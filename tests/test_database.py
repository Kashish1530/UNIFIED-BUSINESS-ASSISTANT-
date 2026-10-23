"""
Focused test suite for Northwind SQLite database verification.
Verifies schema, primary keys, foreign-key integrity, exact row counts, and JOINs.
"""

import sqlite3
import unittest
from src.database.connection import get_database_path

DB_PATH = get_database_path()

EXPECTED_TABLE_COUNTS = {
    "Categories": 8,
    "Customers": 91,
    "Employees": 10,
    "Shippers": 3,
    "Suppliers": 29,
    "Products": 77,
    "Orders": 196,
    "OrderDetails": 518,
}

EXPECTED_PRIMARY_KEYS = {
    "Categories": "CategoryID",
    "Customers": "CustomerID",
    "Employees": "EmployeeID",
    "Shippers": "ShipperID",
    "Suppliers": "SupplierID",
    "Products": "ProductID",
    "Orders": "OrderID",
    "OrderDetails": "OrderDetailID",
}


class TestNorthwindDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Database file not found at {DB_PATH}. Run scripts/convert_northwind.py first.")
        cls.conn = sqlite3.connect(str(DB_PATH))
        cls.conn.execute("PRAGMA foreign_keys = ON;")

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_01_database_file_exists(self):
        """Verify that the database file exists and is not empty."""
        self.assertTrue(DB_PATH.exists(), f"Database file does not exist: {DB_PATH}")
        self.assertGreater(DB_PATH.stat().st_size, 0, "Database file is empty")

    def test_02_all_tables_exist(self):
        """Verify that all eight expected tables are present in the SQLite catalog."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = {row[0] for row in cursor.fetchall()}
        for table in EXPECTED_TABLE_COUNTS.keys():
            self.assertIn(table, tables, f"Expected table '{table}' is missing from database")

    def test_03_primary_keys_configured(self):
        """Verify primary key columns for all eight tables."""
        cursor = self.conn.cursor()
        for table, expected_pk in EXPECTED_PRIMARY_KEYS.items():
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            pk_cols = [col[1] for col in columns if col[5] > 0]
            self.assertEqual(
                pk_cols,
                [expected_pk],
                f"Table {table} expected PK [{expected_pk}], got {pk_cols}",
            )

    def test_04_exact_row_counts(self):
        """Verify that actual imported row counts match expected source counts exactly."""
        cursor = self.conn.cursor()
        total_rows = 0
        for table, expected_count in EXPECTED_TABLE_COUNTS.items():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            actual_count = cursor.fetchone()[0]
            total_rows += actual_count
            self.assertEqual(
                actual_count,
                expected_count,
                f"Row count mismatch for {table}: expected {expected_count}, got {actual_count}",
            )
        self.assertEqual(total_rows, 932, f"Expected 932 total rows across all tables, got {total_rows}")

    def test_05_foreign_key_integrity_check(self):
        """Verify PRAGMA foreign_key_check returns zero errors across all tables."""
        cursor = self.conn.cursor()
        fk_errors = cursor.execute("PRAGMA foreign_key_check").fetchall()
        self.assertEqual(fk_errors, [], f"Foreign key constraint violations found: {fk_errors}")

    def test_06_foreign_key_enforcement_active(self):
        """Verify SQLite actively enforces foreign keys by rejecting invalid parent references."""
        # Use an in-memory duplicate with active foreign keys to test rejection safely without mutating main DB
        test_conn = sqlite3.connect(":memory:")
        test_conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.backup(test_conn)

        with self.assertRaises(sqlite3.IntegrityError):
            # Insert product with non-existent SupplierID (9999) and CategoryID (9999)
            test_conn.execute(
                "INSERT INTO Products (ProductName, SupplierID, CategoryID, Unit, Price) "
                "VALUES ('Invalid FK Item', 9999, 9999, '1 box', 19.99)"
            )
        test_conn.close()

    def test_07_join_products_categories_suppliers(self):
        """Verify representative JOIN between Products, Categories, and Suppliers."""
        cursor = self.conn.cursor()
        query = """
        SELECT p.ProductID, p.ProductName, c.CategoryName, s.SupplierName, p.Price
        FROM Products p
        JOIN Categories c ON p.CategoryID = c.CategoryID
        JOIN Suppliers s ON p.SupplierID = s.SupplierID
        ORDER BY p.ProductID ASC;
        """
        rows = cursor.execute(query).fetchall()
        self.assertEqual(len(rows), 77, "Expected all 77 products to join with Category and Supplier")
        # Spot check first product: Chais, Beverages, Exotic Liquid, 18
        first = rows[0]
        self.assertEqual(first[1], "Chais")
        self.assertEqual(first[2], "Beverages")
        self.assertEqual(first[3], "Exotic Liquid")
        self.assertEqual(first[4], 18)

    def test_08_join_orders_customers_employees_shippers(self):
        """Verify representative JOIN between Orders, Customers, Employees, and Shippers."""
        cursor = self.conn.cursor()
        query = """
        SELECT o.OrderID, c.CustomerName, e.FirstName || ' ' || e.LastName AS EmployeeName, sh.ShipperName
        FROM Orders o
        JOIN Customers c ON o.CustomerID = c.CustomerID
        JOIN Employees e ON o.EmployeeID = e.EmployeeID
        JOIN Shippers sh ON o.ShipperID = sh.ShipperID
        ORDER BY o.OrderID ASC;
        """
        rows = cursor.execute(query).fetchall()
        self.assertEqual(len(rows), 196, "Expected all 196 orders to join with Customer, Employee, and Shipper")
        # Spot check order 10248
        first = rows[0]
        self.assertEqual(first[0], 10248)
        self.assertEqual(first[1], "Wilman Kala")
        self.assertEqual(first[2], "Steven Buchanan")
        self.assertEqual(first[3], "Federal Shipping")

    def test_09_join_order_details_orders_products(self):
        """Verify representative JOIN between OrderDetails, Orders, and Products."""
        cursor = self.conn.cursor()
        query = """
        SELECT od.OrderDetailID, o.OrderID, p.ProductName, od.Quantity, p.Price, (od.Quantity * p.Price) AS Subtotal
        FROM OrderDetails od
        JOIN Orders o ON od.OrderID = o.OrderID
        JOIN Products p ON od.ProductID = p.ProductID
        ORDER BY od.OrderDetailID ASC;
        """
        rows = cursor.execute(query).fetchall()
        self.assertEqual(len(rows), 518, "Expected all 518 order details to join with Orders and Products")
        # Check order detail 1: OrderID 10248, ProductID 11 (Queso Cabrales), Quantity 12, Price 21, Subtotal 252
        first = rows[0]
        self.assertEqual(first[0], 1)
        self.assertEqual(first[1], 10248)
        self.assertEqual(first[2], "Queso Cabrales")
        self.assertEqual(first[3], 12)
        self.assertEqual(first[4], 21)
        self.assertEqual(first[5], 252)


if __name__ == "__main__":
    unittest.main()
