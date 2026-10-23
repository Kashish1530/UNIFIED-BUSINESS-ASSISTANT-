"""
Comprehensive test suite for SQL safety, AST parsing, and validation module.
Verifies single read-only queries, comment/quote handling, unknown tables/columns,
multi-statement rejection, destructive SQL rejection, limit enforcement, and safe execution.
"""

from pathlib import Path
import sqlite3
import unittest

from src.database.connection import ReadOnlyQueryResult, get_database_path
from src.safety.validator import (
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
    SQLSafetyValidator,
    SQLValidationError,
    ValidationResult,
    safe_execute_query,
    validate_sql,
)


class TestSQLSafetyValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = get_database_path()
        cls.validator = SQLSafetyValidator(
            default_limit=100,
            max_limit=1000,
            db_path=cls.db_path,
        )

    # -------------------------------------------------------------------------
    # 1. Valid SELECT Queries & Execution
    # -------------------------------------------------------------------------
    def test_01_valid_basic_select(self):
        """Verify standard SELECT queries pass validation and get default limit applied."""
        res = self.validator.validate("SELECT CustomerID, CustomerName FROM Customers")
        self.assertTrue(res.is_valid, f"Expected valid, got error: {res.error}")
        self.assertIn("LIMIT 100", res.sanitized_sql)
        self.assertIn("Customers", res.tables)

    def test_02_valid_join_queries(self):
        """Verify multi-table joins pass validation."""
        sql = """
        SELECT p.ProductID, p.ProductName, c.CategoryName, s.SupplierName
        FROM Products p
        JOIN Categories c ON p.CategoryID = c.CategoryID
        JOIN Suppliers s ON p.SupplierID = s.SupplierID
        WHERE p.Price > 20
        """
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Expected valid join, got: {res.error}")
        self.assertCountEqual(res.tables, ["Products", "Categories", "Suppliers"])

    def test_03_valid_aggregations_and_grouping(self):
        """Verify aggregations, GROUP BY, HAVING, and ORDER BY pass validation."""
        sql = """
        SELECT Country, COUNT(*) AS TotalCustomers
        FROM Customers
        GROUP BY Country
        HAVING TotalCustomers > 3
        ORDER BY TotalCustomers DESC
        """
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Expected valid aggregation, got: {res.error}")
        self.assertIn("LIMIT 100", res.sanitized_sql)

    def test_04_valid_common_table_expressions(self):
        """Verify CTEs with WITH clause pass validation and local CTE names are recognized."""
        sql = """
        WITH HighValueOrders AS (
            SELECT OrderID, CustomerID FROM Orders WHERE OrderID > 10300
        )
        SELECT h.OrderID, c.CustomerName
        FROM HighValueOrders h
        JOIN Customers c ON h.CustomerID = c.CustomerID
        """
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Expected valid CTE, got: {res.error}")
        self.assertIn("Orders", res.tables)
        self.assertIn("Customers", res.tables)

    def test_05_valid_set_operations_union(self):
        """Verify UNION queries pass validation and outer limit is applied."""
        sql = "SELECT CustomerID FROM Customers UNION SELECT CustomerID FROM Orders"
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Expected valid UNION, got: {res.error}")
        self.assertIn("LIMIT 100", res.sanitized_sql)

    # -------------------------------------------------------------------------
    # 2. Comments, Quoted Identifiers, and String Literals
    # -------------------------------------------------------------------------
    def test_06_comments_handling(self):
        """Verify inline, line, and trailing comments are safely parsed without interfering."""
        sql = """
        -- Leading line comment
        SELECT CategoryID, CategoryName /* inline comment */
        FROM Categories -- trailing comment
        WHERE CategoryID = 1
        """
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Failed comment handling: {res.error}")

    def test_07_quoted_identifiers(self):
        """Verify double quotes and square brackets on identifiers are parsed correctly."""
        sql = 'SELECT "CustomerID", [CustomerName], `City` FROM Customers'
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Failed quoted identifiers: {res.error}")

    def test_08_string_literals_containing_keywords_and_semicolons(self):
        """Verify keywords (DROP, DELETE) and semicolons inside string literals are NOT falsely rejected."""
        # String contains 'Drop St; Apt 4'
        sql = "SELECT * FROM Customers WHERE Address = '123 Drop St; Apt 4'"
        res = self.validator.validate(sql)
        self.assertTrue(res.is_valid, f"Falsely rejected string literal with keyword/semicolon: {res.error}")

        # String with escaped quotes
        sql2 = "SELECT * FROM Customers WHERE CustomerName = 'Bon app'''''"
        res2 = self.validator.validate(sql2)
        self.assertTrue(res2.is_valid, f"Failed escaped quote in string literal: {res2.error}")

    # -------------------------------------------------------------------------
    # 3. Schema & System Table Gating
    # -------------------------------------------------------------------------
    def test_09_unknown_table_rejected(self):
        """Verify queries referencing non-existent tables are rejected."""
        res = self.validator.validate("SELECT * FROM NonExistentTable")
        self.assertFalse(res.is_valid)
        self.assertIn("Unknown or unpermitted table", res.error)

    def test_10_system_tables_rejected(self):
        """Verify sqlite_master and internal tables are rejected."""
        res = self.validator.validate("SELECT * FROM sqlite_master")
        self.assertFalse(res.is_valid)
        self.assertIn("system or pragma object", res.error)

    def test_11_pragma_statements_rejected(self):
        """Verify pragma statements are rejected."""
        res = self.validator.validate("PRAGMA table_info(Customers)")
        self.assertFalse(res.is_valid)
        self.assertTrue("forbidden" in res.error.lower() or "not permitted" in res.error.lower())

    def test_12_unknown_column_rejected(self):
        """Verify queries with invalid columns are caught by SQLite compile check."""
        res = self.validator.validate("SELECT NonExistentColumn FROM Customers")
        self.assertFalse(res.is_valid)
        self.assertIn("no such column", res.error.lower())

    # -------------------------------------------------------------------------
    # 4. Multi-Statement Rejection
    # -------------------------------------------------------------------------
    def test_13_multi_statements_rejected(self):
        """Verify multiple statements separated by semicolons are rejected."""
        res1 = self.validator.validate("SELECT 1; SELECT 2;")
        self.assertFalse(res1.is_valid)
        self.assertIn("Multiple SQL statements are not permitted", res1.error)

        # Valid SELECT followed by destructive command
        res2 = self.validator.validate("SELECT * FROM Customers; DROP TABLE Customers;")
        self.assertFalse(res2.is_valid)
        self.assertIn("Multiple SQL statements are not permitted", res2.error)

        # Trailing mutation after comments
        res3 = self.validator.validate("SELECT * FROM Customers; -- comment\nDELETE FROM Orders;")
        self.assertFalse(res3.is_valid)
        self.assertIn("Multiple SQL statements are not permitted", res3.error)

    # -------------------------------------------------------------------------
    # 5. Destructive & Mutating SQL Rejection
    # -------------------------------------------------------------------------
    def test_14_mutating_and_admin_statements_rejected(self):
        """Verify INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA are rejected."""
        forbidden_statements = [
            ("INSERT INTO Categories (CategoryName) VALUES ('Test')", "Insert"),
            ("UPDATE Customers SET City = 'Metropolis' WHERE CustomerID = 1", "Update"),
            ("DELETE FROM Orders WHERE OrderID = 10248", "Delete"),
            ("DROP TABLE Products", "Drop"),
            ("ALTER TABLE Customers ADD COLUMN Age INT", "Alter"),
            ("CREATE TABLE Malicious (id INT)", "Create"),
            ("ATTACH DATABASE 'malicious.db' AS mal", "Attach"),
            ("DETACH DATABASE mal", "Detach"),
            ("PRAGMA foreign_keys = OFF", "Pragma"),
        ]
        for sql, expected_type in forbidden_statements:
            res = self.validator.validate(sql)
            self.assertFalse(res.is_valid, f"Failed to reject: {sql}")
            self.assertTrue(
                "forbidden" in res.error.lower() or "permitted" in res.error.lower(),
                f"Unexpected error message for {sql}: {res.error}",
            )

    def test_15_forbidden_function_calls_rejected(self):
        """Verify dangerous function calls like load_extension are rejected."""
        res = self.validator.validate("SELECT load_extension('malicious.so')")
        self.assertFalse(res.is_valid)
        self.assertIn("load_extension", res.error)

    # -------------------------------------------------------------------------
    # 6. Malformed Queries
    # -------------------------------------------------------------------------
    def test_16_malformed_syntax_and_empty_queries(self):
        """Verify syntax errors and empty strings are rejected."""
        res_empty = self.validator.validate("")
        self.assertFalse(res_empty.is_valid)
        self.assertIn("empty", res_empty.error.lower())

        res_ws = self.validator.validate("   \n\t  ")
        self.assertFalse(res_ws.is_valid)
        self.assertIn("empty", res_ws.error.lower())

        res_syntax = self.validator.validate("SELECT FROM WHERE")
        self.assertFalse(res_syntax.is_valid)
        self.assertIn("syntax error", res_syntax.error.lower())

        res_unclosed = self.validator.validate("SELECT * FROM Customers WHERE (CustomerID = 1")
        self.assertFalse(res_unclosed.is_valid)
        self.assertIn("syntax error", res_unclosed.error.lower())

    # -------------------------------------------------------------------------
    # 7. Row Limit Enforcement
    # -------------------------------------------------------------------------
    def test_17_row_limit_enforcement_behavior(self):
        """Verify LIMIT is injected or capped without altering query meaning."""
        # 1. No limit -> inject default (100)
        s1, _ = self.validator.validate_and_sanitize("SELECT * FROM Customers")
        self.assertIn("LIMIT 100", s1)

        # 2. Existing limit below max -> preserve (15)
        s2, _ = self.validator.validate_and_sanitize("SELECT * FROM Customers LIMIT 15")
        self.assertIn("LIMIT 15", s2)

        # 3. Existing limit above max (5000 > 1000) -> capped to max (1000)
        s3, _ = self.validator.validate_and_sanitize("SELECT * FROM Customers LIMIT 5000")
        self.assertIn("LIMIT 1000", s3)

        # 4. Limit on query with subquery -> subquery limit is preserved, outer gets default
        s4, _ = self.validator.validate_and_sanitize(
            "SELECT * FROM Customers WHERE CustomerID IN (SELECT CustomerID FROM Orders LIMIT 5)"
        )
        self.assertIn("LIMIT 5", s4)
        self.assertIn("LIMIT 100", s4)

    # -------------------------------------------------------------------------
    # 8. Safe Execution Integration
    # -------------------------------------------------------------------------
    def test_18_safe_execute_query_success(self):
        """Verify safe_execute_query executes valid queries through read-only layer."""
        result = safe_execute_query(
            "SELECT ProductID, ProductName, Price FROM Products WHERE Price > 50 ORDER BY Price DESC"
        )
        self.assertIsInstance(result, ReadOnlyQueryResult)
        self.assertEqual(result.columns, ["ProductID", "ProductName", "Price"])
        self.assertGreater(result.row_count, 0)
        for row in result.rows:
            self.assertGreater(row["Price"], 50)

    def test_19_safe_execute_zero_row_query(self):
        """Verify zero-row query executes through safe_execute_query as a valid result."""
        result = safe_execute_query("SELECT CustomerID, CustomerName FROM Customers WHERE CustomerID = -9999")
        self.assertEqual(result.columns, ["CustomerID", "CustomerName"])
        self.assertEqual(result.rows, [])
        self.assertEqual(result.row_count, 0)

    def test_20_safe_execute_blocks_mutation(self):
        """Verify safe_execute_query raises SQLValidationError on destructive input without executing."""
        with self.assertRaises(SQLValidationError) as ctx:
            safe_execute_query("DROP TABLE Categories")
        self.assertIn("forbidden", str(ctx.exception).lower())

        # Verify Categories table still exists and has 8 rows
        check = safe_execute_query("SELECT COUNT(*) AS cnt FROM Categories")
        self.assertEqual(check.rows[0]["cnt"], 8)


if __name__ == "__main__":
    unittest.main()
