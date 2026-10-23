"""
Northwind Neon PostgreSQL Verification Script
=============================================
Read-only: connects to Neon and checks row counts and FK JOIN integrity.
Never prints, logs, or exposes the connection string.

Usage
-----
    python scripts/verify_postgres.py
"""

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected state (mirrors verified SQLite baseline)
# ---------------------------------------------------------------------------

EXPECTED_COUNTS: dict[str, int] = {
    "categories":   8,
    "customers":    91,
    "employees":    10,
    "shippers":     3,
    "suppliers":    29,
    "products":     77,
    "orders":       196,
    "orderdetails": 518,
}

TOTAL_EXPECTED = sum(EXPECTED_COUNTS.values())  # 932

# Representative FK JOIN checks: (description, sql, expected_row_count)
FK_CHECKS: list[tuple[str, str, int]] = [
    (
        "Orders -> Customers -> Employees -> Shippers",
        """
        SELECT COUNT(*) FROM orders o
        JOIN customers c  ON c.customerid  = o.customerid
        JOIN employees e  ON e.employeeid  = o.employeeid
        JOIN shippers  s  ON s.shipperid   = o.shipperid
        """,
        196,
    ),
    (
        "OrderDetails -> Orders -> Products",
        """
        SELECT COUNT(*) FROM orderdetails od
        JOIN orders   o ON o.orderid   = od.orderid
        JOIN products p ON p.productid = od.productid
        """,
        518,
    ),
    (
        "Products -> Categories",
        """
        SELECT COUNT(*) FROM products p
        JOIN categories c ON c.categoryid = p.categoryid
        """,
        77,
    ),
    (
        "Products -> Suppliers",
        """
        SELECT COUNT(*) FROM products p
        JOIN suppliers s ON s.supplierid = p.supplierid
        """,
        77,
    ),
]

# Sequence sanity checks: (table, pk_col, min_expected_next_val)
# Orders start at 10248 so the sequence must be at least 10443.
SEQUENCE_CHECKS: list[tuple[str, str, int]] = [
    ("categories",   "categoryid",   8),
    ("customers",    "customerid",   91),
    ("employees",    "employeeid",   10),
    ("shippers",     "shipperid",    3),
    ("suppliers",    "supplierid",   29),
    ("products",     "productid",    77),
    ("orders",       "orderid",      10443),
    ("orderdetails", "orderdetailid", 518),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env_file(env_path: Path | None = None) -> None:
    target = env_path or (get_project_root() / ".env")
    if not target.exists():
        return
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = val


def get_neon_url() -> str:
    url = os.environ.get("NEONDB_URL", "").strip()
    if not url:
        raise RuntimeError(
            "NEONDB_URL is not set. Add it to .env or set it as an environment variable."
        )
    return url


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification() -> None:
    load_env_file()
    neon_url = get_neon_url()

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2-binary is not installed.\nRun: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    print("Connecting to Neon PostgreSQL (read-only checks)...")
    # URL never printed
    pg_conn = psycopg2.connect(neon_url)
    pg_conn.autocommit = True          # read-only; no transaction needed
    pg_cur = pg_conn.cursor()

    # Enforce read-only session characteristics
    pg_cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")

    failures: list[str] = []

    try:
        # ----------------------------------------------------------------
        # 1. Row count verification
        # ----------------------------------------------------------------
        print("\n[1/3] Row count verification")
        print(f"  {'Table':<15}  {'Expected':>8}  {'Actual':>8}  {'Status'}")
        print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*6}")

        total_actual = 0
        for tbl, expected in EXPECTED_COUNTS.items():
            pg_cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            actual = pg_cur.fetchone()[0]
            total_actual += actual
            ok = actual == expected
            status = "OK" if ok else "FAIL"
            print(f"  {tbl:<15}  {expected:>8}  {actual:>8}  [{status}]")
            if not ok:
                failures.append(f"Row count mismatch in {tbl}: expected {expected}, got {actual}")

        total_ok = total_actual == TOTAL_EXPECTED
        print(f"  {'TOTAL':<15}  {TOTAL_EXPECTED:>8}  {total_actual:>8}  [{'OK' if total_ok else 'FAIL'}]")
        if not total_ok:
            failures.append(f"Total row count mismatch: expected {TOTAL_EXPECTED}, got {total_actual}")

        # ----------------------------------------------------------------
        # 2. FK JOIN integrity checks
        # ----------------------------------------------------------------
        print("\n[2/3] Foreign-key JOIN integrity checks")
        for desc, sql, expected_count in FK_CHECKS:
            pg_cur.execute(sql)
            actual = pg_cur.fetchone()[0]
            ok = actual == expected_count
            status = "OK" if ok else "FAIL"
            print(f"  {status}  {desc}  ({actual} rows, expected {expected_count})")
            if not ok:
                failures.append(f"FK JOIN check failed [{desc}]: expected {expected_count}, got {actual}")

        # ----------------------------------------------------------------
        # 3. Sequence sanity checks
        # ----------------------------------------------------------------
        print("\n[3/3] Identity sequence checks")
        from psycopg2 import sql

        for tbl, pk, min_next in SEQUENCE_CHECKS:
            # Query the actual maximum PK present in the table
            pg_cur.execute(
                sql.SQL("SELECT COALESCE(MAX({}), 0) FROM {}").format(
                    sql.Identifier(pk), sql.Identifier(tbl)
                )
            )
            max_pk = pg_cur.fetchone()[0]

            # 1. Retrieve the sequence schema, sequence name, and increment_by safely from catalogs
            # Explicitly target the sequence for public.<tbl>.<pk>
            pg_cur.execute(
                """
                SELECT s.schemaname, s.sequencename, s.increment_by
                FROM pg_sequences s
                JOIN pg_class c ON c.relname = s.sequencename
                JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = s.schemaname
                WHERE c.oid = pg_get_serial_sequence('public.' || %s, %s)::regclass
                  AND s.schemaname = 'public'
                """,
                (tbl, pk),
            )
            seq_meta = pg_cur.fetchone()

            if not seq_meta or not seq_meta[0] or not seq_meta[1]:
                failures.append(
                    f"Unable to find sequence for {tbl}.{pk} in catalogs"
                )
                print(f"  FAIL  {tbl}.{pk}: sequence not found in pg_sequences")
                continue

            seq_schema, seq_name, increment_by = seq_meta[0], seq_meta[1], seq_meta[2]
            if increment_by is None:
                increment_by = 1

            # 2. Inspect last_value and is_called directly from the actual sequence relation
            try:
                pg_cur.execute(
                    sql.SQL("SELECT last_value, is_called FROM {}.{}").format(
                        sql.Identifier(seq_schema), sql.Identifier(seq_name)
                    )
                )
                seq_row = pg_cur.fetchone()
            except Exception as e:
                failures.append(
                    f"Error reading sequence relation {seq_schema}.{seq_name} for {tbl}.{pk}: {e}"
                )
                print(f"  FAIL  {tbl}.{pk}: failed to query sequence relation ({e})")
                continue

            if not seq_row or seq_row[0] is None or seq_row[1] is None:
                failures.append(
                    f"Unable to determine sequence state for {tbl}.{pk} (NULL returned from sequence relation)"
                )
                print(f"  FAIL  {tbl}.{pk}: sequence relation returned NULL values")
                continue

            last_val, is_called = seq_row[0], seq_row[1]

            # Next generated value:
            # If is_called=True, next value is last_value + increment_by.
            # If is_called=False, next value is last_value.
            next_val = (last_val + increment_by) if is_called else last_val

            # Sequence must not collide with max_pk (next_val > max_pk)
            # and the next generated value must meet the expected threshold (next_val >= min_next)
            no_collision = next_val > max_pk if max_pk > 0 else True
            meets_min = next_val >= min_next
            ok = no_collision and meets_min
            status = "OK" if ok else "FAIL"

            print(
                f"  {status}  {tbl}.{pk}: last_value={last_val}, is_called={is_called}, "
                f"next_val={next_val}, max_pk={max_pk} (min expected next {min_next})"
            )
            if not ok:
                if not no_collision:
                    failures.append(
                        f"Sequence collision risk for {tbl}.{pk}: next generated value {next_val} <= max primary key {max_pk}"
                    )
                if not meets_min:
                    failures.append(
                        f"Sequence too low for {tbl}.{pk}: next generated value {next_val} < min expected {min_next}"
                    )

    finally:
        pg_cur.close()
        pg_conn.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"[FAIL] VERIFICATION FAILED - {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("[OK] All verification checks passed.")
        print(f"  {TOTAL_EXPECTED} rows confirmed across 8 tables.")
        print("  Foreign-key JOIN integrity: OK")
        print("  Identity sequences: OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_verification()
