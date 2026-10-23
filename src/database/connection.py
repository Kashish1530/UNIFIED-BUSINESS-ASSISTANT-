"""
Database connection and read-only execution module for Role B Data Service.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Sequence

DEFAULT_TIMEOUT = 5.0


def get_project_root() -> Path:
    """
    Resolve the project root directory reliably relative to this file location,
    independent of the caller's current working directory.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # src/database/connection.py -> parent is database, parent.parent is src, parent.parent.parent is project root
    return Path(__file__).resolve().parent.parent.parent


def get_database_path(custom_path: Path | str | None = None) -> Path:
    """
    Resolve database path reliably.
    Precedence:
    1. Explicit custom_path argument
    2. NORTHWIND_DB_PATH environment variable
    3. <project_root>/data/northwind.db
    """
    if custom_path:
        p = Path(custom_path)
        if not p.is_absolute():
            return (get_project_root() / p).resolve()
        return p.resolve()

    env_path = os.environ.get("NORTHWIND_DB_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            return (get_project_root() / p).resolve()
        return p.resolve()

    return (get_project_root() / "data" / "northwind.db").resolve()


def get_read_only_connection(
    db_path: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> sqlite3.Connection:
    """
    Open and configure a read-only SQLite database connection.
    Enforces and verifies:
    - PRAGMA foreign_keys = ON
    - PRAGMA query_only = ON
    - Row factory configured to sqlite3.Row
    - Dual-layer protection: VFS mode=ro where applicable and query_only PRAGMA.
    """
    target = get_database_path(custom_path=db_path) if db_path != ":memory:" else ":memory:"

    if target != ":memory:":
        target_path = Path(target)
        if not target_path.exists():
            raise FileNotFoundError(f"Database file not found at: {target_path}")
        # Use URI mode=ro for OS/VFS-level read-only protection
        uri = f"{target_path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(":memory:", timeout=timeout)

    try:
        conn.row_factory = sqlite3.Row

        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        fk_status = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        if fk_status != 1:
            raise RuntimeError("Failed to enable PRAGMA foreign_keys.")

        # Enforce read-only queries
        conn.execute("PRAGMA query_only = ON;")
        qo_status = conn.execute("PRAGMA query_only;").fetchone()[0]
        if qo_status != 1:
            raise RuntimeError("Failed to enable PRAGMA query_only.")

        return conn
    except Exception:
        conn.close()
        raise


@contextmanager
def read_only_connection(
    db_path: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[sqlite3.Connection]:
    """
    Context manager that yields a read-only SQLite connection
    and guarantees reliable connection closure upon exit or exception.
    """
    conn = get_read_only_connection(db_path=db_path, timeout=timeout)
    try:
        yield conn
    finally:
        conn.close()


@dataclass(frozen=True)
class ReadOnlyQueryResult:
    """
    Result container for read-only database query execution.
    Attributes:
        columns: List of column names returned by query.
        rows: List of sqlite3.Row objects.
    """
    columns: list[str]
    rows: list[sqlite3.Row]

    def __iter__(self):
        """Allows tuple unpacking: columns, rows = execute_read_only_query(...)"""
        return iter((self.columns, self.rows))

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convenience helper to convert rows to dictionary format."""
        return [dict(r) for r in self.rows]


def execute_read_only_query(
    query: str,
    params: Sequence[Any] | dict[str, Any] | None = None,
    db_path: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ReadOnlyQueryResult:
    """
    Execute a read-only query safely.
    Guarantees:
    - Connection is closed reliably in all circumstances (success or exception).
    - Query cannot mutate data (rejected by PRAGMA query_only / VFS).
    - Returns column names and row objects.
    - Zero rows returned is handled gracefully as a valid result.
    """
    with read_only_connection(db_path=db_path, timeout=timeout) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return ReadOnlyQueryResult(columns=columns, rows=rows)
