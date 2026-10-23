"""
Focused test suite for database connection and read-only execution module.
Verifies path resolution, timeout, pragmas, SELECT execution, zero-row results,
connection cleanup, and strict rejection of mutating/schema-changing statements.
"""

from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from src.database.connection import (
    DEFAULT_TIMEOUT,
    ReadOnlyQueryResult,
    execute_read_only_query,
    get_database_path,
    get_project_root,
    get_read_only_connection,
    read_only_connection,
)


class TestDatabaseConnection(unittest.TestCase):
    def test_01_get_project_root_resolution(self):
        """Verify project root resolves to the folder containing data/ and scripts/."""
        root = get_project_root()
        self.assertTrue(root.exists(), f"Project root does not exist: {root}")
        self.assertTrue((root / "data" / "northwind.db").exists(), "northwind.db not found at project root")
        self.assertTrue((root / "scripts" / "convert_northwind.py").exists(), "convert_northwind.py not found")

    def test_02_get_database_path_defaults(self):
        """Verify get_database_path resolves default database path reliably."""
        db_path = get_database_path()
        self.assertTrue(db_path.is_absolute(), f"Path is not absolute: {db_path}")
        self.assertTrue(db_path.exists(), f"Resolved database does not exist: {db_path}")
        self.assertEqual(db_path.name, "northwind.db")

    def test_03_get_database_path_custom_and_env(self):
        """Verify custom path and env variable resolution."""
        # Custom relative path resolves against project root
        custom_rel = get_database_path("data/northwind.db")
        self.assertTrue(custom_rel.is_absolute())
        self.assertEqual(custom_rel, get_database_path())

        # Environment variable override
        with patch.dict("os.environ", {"NORTHWIND_DB_PATH": "data/northwind.db"}):
            env_db = get_database_path()
            self.assertEqual(env_db, get_database_path())

    def test_04_connection_pragmas_and_row_factory(self):
        """Verify foreign_keys=ON, query_only=ON, and sqlite3.Row row_factory."""
        conn = get_read_only_connection()
        try:
            # Check row factory
            self.assertEqual(conn.row_factory, sqlite3.Row)

            # Check foreign keys
            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            self.assertEqual(fk, 1, "PRAGMA foreign_keys is not ON")

            # Check query_only
            qo = conn.execute("PRAGMA query_only;").fetchone()[0]
            self.assertEqual(qo, 1, "PRAGMA query_only is not ON")
        finally:
            conn.close()

    def test_05_valid_select_query_execution(self):
        """Verify read-only execution returns correct columns and row objects."""
        result = execute_read_only_query("SELECT CategoryID, CategoryName FROM Categories ORDER BY CategoryID ASC")
        self.assertIsInstance(result, ReadOnlyQueryResult)
        self.assertEqual(result.columns, ["CategoryID", "CategoryName"])
        self.assertEqual(result.row_count, 8)

        # Verify tuple unpacking
        cols, rows = result
        self.assertEqual(cols, ["CategoryID", "CategoryName"])
        self.assertEqual(len(rows), 8)

        # Verify row object features: index access and name access
        first_row = rows[0]
        self.assertEqual(first_row[0], 1)
        self.assertEqual(first_row["CategoryID"], 1)
        self.assertEqual(first_row["CategoryName"], "Beverages")

        # Verify to_dicts helper
        dicts = result.to_dicts()
        self.assertEqual(len(dicts), 8)
        self.assertEqual(dicts[0], {"CategoryID": 1, "CategoryName": "Beverages"})

    def test_06_parameterized_query(self):
        """Verify parameters are correctly passed to read-only queries."""
        result = execute_read_only_query(
            "SELECT CustomerID, CustomerName, Country FROM Customers WHERE Country = ? ORDER BY CustomerID ASC",
            params=("Germany",),
        )
        self.assertEqual(result.columns, ["CustomerID", "CustomerName", "Country"])
        self.assertEqual(result.row_count, 11)
        for row in result.rows:
            self.assertEqual(row["Country"], "Germany")

    def test_07_zero_row_query_handled_validly(self):
        """Verify queries returning 0 rows are valid results, not errors."""
        result = execute_read_only_query(
            "SELECT CustomerID, CustomerName, City FROM Customers WHERE CustomerID = -9999"
        )
        self.assertIsInstance(result, ReadOnlyQueryResult)
        # Column names should still be present from cursor description
        self.assertEqual(result.columns, ["CustomerID", "CustomerName", "City"])
        self.assertEqual(result.rows, [])
        self.assertEqual(result.row_count, 0)
        self.assertEqual(result.to_dicts(), [])

    def test_08_rejection_of_insert_mutations(self):
        """Verify INSERT is blocked by read-only connection and leaves database unmodified."""
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query(
                "INSERT INTO Categories (CategoryName, Description) VALUES ('Sneaky Cat', 'Should Fail')"
            )
        self.assertIn("readonly", str(ctx.exception).lower())

        # Verify count did not change
        check = execute_read_only_query("SELECT COUNT(*) AS total FROM Categories")
        self.assertEqual(check.rows[0]["total"], 8)

    def test_09_rejection_of_update_mutations(self):
        """Verify UPDATE is blocked by read-only connection."""
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query("UPDATE Categories SET CategoryName = 'Hacked' WHERE CategoryID = 1")
        self.assertIn("readonly", str(ctx.exception).lower())

        # Verify data unchanged
        check = execute_read_only_query("SELECT CategoryName FROM Categories WHERE CategoryID = 1")
        self.assertEqual(check.rows[0]["CategoryName"], "Beverages")

    def test_10_rejection_of_delete_mutations(self):
        """Verify DELETE is blocked by read-only connection."""
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query("DELETE FROM Categories WHERE CategoryID = 1")
        self.assertIn("readonly", str(ctx.exception).lower())

        # Verify data unchanged
        check = execute_read_only_query("SELECT COUNT(*) AS total FROM Categories")
        self.assertEqual(check.rows[0]["total"], 8)

    def test_11_rejection_of_schema_mutations(self):
        """Verify CREATE, DROP, and ALTER statements are blocked."""
        # CREATE TABLE
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query("CREATE TABLE malicious_table (id INT)")
        self.assertIn("readonly", str(ctx.exception).lower())

        # DROP TABLE
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query("DROP TABLE Categories")
        self.assertIn("readonly", str(ctx.exception).lower())

        # ALTER TABLE
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            execute_read_only_query("ALTER TABLE Categories ADD COLUMN extra_col TEXT")
        self.assertIn("readonly", str(ctx.exception).lower())

    def test_12_reliable_connection_closure_on_success_and_exception(self):
        """Verify connection is always closed, even when query execution raises an exception."""
        # 1. Successful execution
        with read_only_connection() as conn:
            self.assertIsNotNone(conn)
        # Outside context, connection should be closed: executing fails with ProgrammingError
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

        # 2. Exception during execution
        saved_conn = None
        try:
            with read_only_connection() as conn:
                saved_conn = conn
                # Intentionally invalid SQL syntax
                conn.execute("THIS IS NOT VALID SQL")
        except sqlite3.OperationalError:
            pass

        self.assertIsNotNone(saved_conn)
        with self.assertRaises(sqlite3.ProgrammingError):
            saved_conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
