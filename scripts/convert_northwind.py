"""
Northwind SQLite Conversion & Import Script.
Converts MySQL Northwind DDL/DML to standard SQLite with foreign key support.
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

EXPECTED_TABLES = [
    "Categories",
    "Customers",
    "Employees",
    "Shippers",
    "Suppliers",
    "Products",
    "Orders",
    "OrderDetails",
]


def count_source_inserts(source_sql: str) -> Counter:
    """Count INSERT INTO statements per table in source script."""
    inserts = re.findall(r"INSERT\s+INTO\s+(\w+)", source_sql, flags=re.IGNORECASE)
    # Normalize table names
    normalized = []
    lookup = {t.lower(): t for t in EXPECTED_TABLES}
    for item in inserts:
        normalized.append(lookup.get(item.lower(), item))
    return Counter(normalized)


def adapt_mysql_to_sqlite(source_sql: str) -> str:
    """
    Carefully adapt MySQL-specific syntax to SQLite dialect.
    Avoids naive/blind replacements (e.g. matching 'USE' inside 'House').
    """
    sql = source_sql

    # 1. Remove CREATE DATABASE statements (anchored to statement start)
    sql = re.sub(
        r"^\s*CREATE\s+DATABASE[^;]+;",
        "",
        sql,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 2. Remove USE statements (anchored to line start to avoid matching words like 'House')
    sql = re.sub(
        r"^\s*USE\s+[^;]+;",
        "",
        sql,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 3. Convert START TRANSACTION to BEGIN TRANSACTION
    sql = re.sub(
        r"^\s*START\s+TRANSACTION;",
        "BEGIN TRANSACTION;",
        sql,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # 4. Convert AUTO_INCREMENT to AUTOINCREMENT (word boundary check)
    sql = re.sub(
        r"\bAUTO_INCREMENT\b",
        "AUTOINCREMENT",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


def inspect_existing_db(db_path: Path):
    """Inspect tables and row counts in an existing SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        print(f"Existing database found at: {db_path}")
        print(f"Tables present ({len(tables)}): {', '.join(tables)}")
        for t in tables:
            cnt = cursor.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  - {t}: {cnt} rows")
        conn.close()
    except Exception as e:
        print(f"Error inspecting existing database {db_path}: {e}")


def convert_and_import(source_path: Path, target_path: Path, force: bool = False) -> bool:
    """Run the conversion and import process."""
    if not source_path.exists():
        print(f"ERROR: Source file does not exist: {source_path}", file=sys.stderr)
        return False

    if target_path.exists():
        if not force:
            print(f"WARNING: Target database already exists: {target_path}")
            print("To prevent accidental data loss or duplication, conversion was skipped.")
            print("Use --force to overwrite the existing database.")
            print("\nCurrent database contents:")
            inspect_existing_db(target_path)
            return False
        else:
            print(f"Notice: --force specified. Overwriting existing database: {target_path}")
            target_path.unlink()

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading source SQL from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        source_sql = f.read()

    expected_counts = count_source_inserts(source_sql)
    print("Parsed source INSERT counts:")
    for t in EXPECTED_TABLES:
        print(f"  - {t}: {expected_counts.get(t, 0)} expected rows")
    total_expected = sum(expected_counts.values())
    print(f"  Total expected rows: {total_expected}")

    print("\nAdapting SQL syntax to SQLite dialect...")
    sqlite_sql = adapt_mysql_to_sqlite(source_sql)

    # Use a temporary file for atomic creation
    temp_target = target_path.with_suffix(".tmp.db")
    if temp_target.exists():
        temp_target.unlink()

    print(f"Creating SQLite database at: {temp_target}")
    conn = sqlite3.connect(str(temp_target))
    try:
        # Enable foreign key constraint checking
        conn.execute("PRAGMA foreign_keys = ON;")

        # Execute adapted script
        conn.executescript(sqlite_sql)
        conn.commit()

        # Check foreign keys
        print("\nVerifying foreign key constraints...")
        fk_errors = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if fk_errors:
            print(f"ERROR: Foreign key check failed with {len(fk_errors)} errors:", file=sys.stderr)
            for err in fk_errors:
                print(f"  Table: {err[0]}, RowId: {err[1]}, Target: {err[2]}, FkId: {err[3]}", file=sys.stderr)
            conn.close()
            temp_target.unlink()
            return False
        print("  Foreign key check passed: 0 integrity errors.")

        # Verify actual row counts
        print("\nVerifying actual imported row counts:")
        mismatches = []
        actual_counts = {}
        for t in EXPECTED_TABLES:
            actual = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            actual_counts[t] = actual
            expected = expected_counts.get(t, 0)
            status = "MATCH" if actual == expected else "MISMATCH"
            print(f"  - {t:<15}: {actual:>4} rows (expected {expected:>4}) -> {status}")
            if actual != expected:
                mismatches.append((t, expected, actual))

        if mismatches:
            print(f"ERROR: Row count mismatches detected: {mismatches}", file=sys.stderr)
            conn.close()
            temp_target.unlink()
            return False

        conn.close()

        # Atomically replace or move into target
        if temp_target.exists():
            temp_target.replace(target_path)

        print(f"\nSUCCESS: Database successfully built and verified at: {target_path}")
        print(f"Total rows imported: {sum(actual_counts.values())}")
        return True

    except Exception as e:
        print(f"ERROR during conversion/import: {e}", file=sys.stderr)
        conn.close()
        if temp_target.exists():
            temp_target.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Northwind MySQL script to SQLite.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/northwind.sql"),
        help="Path to source MySQL script (default: data/northwind.sql)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/northwind.db"),
        help="Path to target SQLite DB (default: data/northwind.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite target database if it already exists",
    )
    args = parser.parse_args()

    success = convert_and_import(args.source, args.target, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
