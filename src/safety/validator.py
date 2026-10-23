"""
SQL Safety, AST Parsing, and Validation Engine for Role B Data Service.
Enforces single read-only queries, blocks mutations/system access, validates
tables/columns, and applies result-row limits without breaking query semantics.
"""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Sequence

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from src.database.connection import (
    DEFAULT_TIMEOUT,
    ReadOnlyQueryResult,
    execute_read_only_query,
    get_read_only_connection,
)

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 1000

# Canonical Northwind SQLite tables
NORTHWIND_TABLES = {
    "categories",
    "customers",
    "employees",
    "shippers",
    "suppliers",
    "products",
    "orders",
    "orderdetails",
}

# Forbidden AST statement types that mutate data, schema, or system state
FORBIDDEN_ROOT_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

# Forbidden function calls that could compromise environment
FORBIDDEN_FUNCTIONS = {
    "load_extension",
    "eval",
    "exec",
}


class SQLValidationError(Exception):
    """Raised when a candidate query fails safety, syntax, or policy validation."""
    pass


@dataclass(frozen=True)
class ValidationResult:
    """Result object of SQL validation."""
    is_valid: bool
    sanitized_sql: str | None
    error: str | None
    tables: list[str]


class SQLSafetyValidator:
    """
    Validates candidate SQL queries for safe, bounded, read-only execution.
    """

    def __init__(
        self,
        default_limit: int = DEFAULT_ROW_LIMIT,
        max_limit: int = MAX_ROW_LIMIT,
        allowed_tables: set[str] | None = None,
        db_path: Path | str | None = None,
    ):
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.allowed_tables = allowed_tables or NORTHWIND_TABLES
        self.db_path = db_path

    def _parse_and_check_multi_statements(self, raw_sql: str) -> exp.Expression:
        """Parse SQL and ensure exactly one statement is present."""
        if not raw_sql or not raw_sql.strip():
            raise SQLValidationError("Query cannot be empty.")

        try:
            parsed_list = sqlglot.parse(raw_sql, read="sqlite")
        except ParseError as e:
            raise SQLValidationError(f"SQL syntax error: {e}")

        # Filter out comments and trailing empty semicolons
        meaningful = [
            s for s in parsed_list
            if s is not None and not isinstance(s, exp.Semicolon)
        ]

        if not meaningful:
            raise SQLValidationError("Query contains no executable statements.")

        if len(meaningful) > 1:
            types = [type(s).__name__ for s in meaningful]
            raise SQLValidationError(
                f"Multiple SQL statements are not permitted ({len(meaningful)} detected: {', '.join(types)})."
            )

        return meaningful[0]

    def _verify_statement_type(self, expression: exp.Expression):
        """Verify that statement is read-only SELECT or SetOperation."""
        if isinstance(expression, FORBIDDEN_ROOT_TYPES):
            type_name = type(expression).__name__
            raise SQLValidationError(
                f"Statement type '{type_name}' is forbidden. Only single read-only SELECT queries are allowed."
            )

        if not isinstance(expression, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            type_name = type(expression).__name__
            raise SQLValidationError(
                f"Statement type '{type_name}' is not permitted. Only single read-only SELECT queries are allowed."
            )

        # Walk entire AST to detect any nested mutations or forbidden subnodes
        for node in expression.walk():
            if isinstance(node, FORBIDDEN_ROOT_TYPES):
                type_name = type(node).__name__
                raise SQLValidationError(f"Forbidden construct '{type_name}' detected inside query.")

            if isinstance(node, (exp.Anonymous, exp.Func)):
                func_name = getattr(node, "name", "").lower()
                if func_name in FORBIDDEN_FUNCTIONS:
                    raise SQLValidationError(f"Function call '{func_name}' is strictly prohibited.")

    def _validate_tables(self, expression: exp.Expression) -> list[str]:
        """Validate referenced tables against approved Northwind schema and CTEs."""
        # Find all CTE names declared in the query
        ctes = {
            cte.alias_or_name.lower()
            for cte in expression.find_all(exp.CTE)
            if cte.alias_or_name
        }

        tables_found: list[str] = []
        for t in expression.find_all(exp.Table):
            table_name = t.name
            if not table_name:
                continue

            lower_name = table_name.lower()

            # Local CTE references are permitted
            if lower_name in ctes:
                continue

            # Block system tables and pragma table-valued functions
            if lower_name.startswith("pragma_") or lower_name.startswith("sqlite_"):
                raise SQLValidationError(f"Access to system or pragma object '{table_name}' is not permitted.")

            if lower_name not in self.allowed_tables:
                permitted = ", ".join(sorted([tbl.capitalize() for tbl in self.allowed_tables]))
                raise SQLValidationError(
                    f"Unknown or unpermitted table '{table_name}'. Permitted tables are: {permitted}."
                )

            if table_name not in tables_found:
                tables_found.append(table_name)

        return tables_found

    def _enforce_row_limit(self, expression: exp.Expression) -> exp.Expression:
        """
        Enforce configurable result-row limit on the outer query without breaking subqueries.
        Note: Queries without ORDER BY will produce non-deterministic row selection when limited.
        """
        existing_limit = expression.args.get("limit")

        if existing_limit is None:
            expression.set("limit", exp.Limit(expression=exp.Literal.number(self.default_limit)))
        else:
            try:
                limit_expr = existing_limit.expression
                val = int(limit_expr.this) if hasattr(limit_expr, "this") else None
                if val is not None and val > self.max_limit:
                    expression.set("limit", exp.Limit(expression=exp.Literal.number(self.max_limit)))
            except Exception:
                # If existing limit expression is non-numeric, cap it to default limit
                expression.set("limit", exp.Limit(expression=exp.Literal.number(self.default_limit)))

        return expression

    def _compile_check_with_sqlite(self, sanitized_sql: str):
        """
        Compiler check: Runs EXPLAIN on the read-only connection with authorizer enabled
        to verify that columns, tables, and functions resolve in the actual database catalog.
        """
        conn = get_read_only_connection(db_path=self.db_path)

        def authorizer(action, arg1, arg2, dbname, source):
            # Authorize only select, read, and function evaluation
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        try:
            conn.set_authorizer(authorizer)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sanitized_sql}")
            cursor.fetchall()
        except sqlite3.OperationalError as e:
            raise SQLValidationError(f"Database validation error: {e}")
        finally:
            conn.close()

    def validate_and_sanitize(self, raw_sql: str) -> tuple[str, list[str]]:
        """
        Validate candidate SQL and return (sanitized_sql, tables_referenced).
        Raises SQLValidationError if query violates any safety or schema rule.
        """
        expression = self._parse_and_check_multi_statements(raw_sql)
        self._verify_statement_type(expression)
        tables = self._validate_tables(expression)
        limited_expression = self._enforce_row_limit(expression)
        sanitized_sql = limited_expression.sql(dialect="sqlite")

        # Compile check against database
        self._compile_check_with_sqlite(sanitized_sql)

        return sanitized_sql, tables

    def validate(self, raw_sql: str) -> ValidationResult:
        """Non-raising validation helper returning a ValidationResult."""
        try:
            sanitized_sql, tables = self.validate_and_sanitize(raw_sql)
            return ValidationResult(
                is_valid=True,
                sanitized_sql=sanitized_sql,
                error=None,
                tables=tables,
            )
        except SQLValidationError as e:
            return ValidationResult(
                is_valid=False,
                sanitized_sql=None,
                error=str(e),
                tables=[],
            )

    def execute(
        self,
        raw_sql: str,
        params: Sequence[Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> ReadOnlyQueryResult:
        """
        Validate, sanitize, and safely execute query via read-only connection layer.
        """
        sanitized_sql, _ = self.validate_and_sanitize(raw_sql)
        return execute_read_only_query(
            query=sanitized_sql,
            params=params,
            db_path=self.db_path,
            timeout=timeout,
        )


# Global default validator instance
_default_validator = SQLSafetyValidator()


def validate_sql(
    raw_sql: str,
    default_limit: int = DEFAULT_ROW_LIMIT,
    max_limit: int = MAX_ROW_LIMIT,
) -> ValidationResult:
    """Convenience function to validate candidate SQL."""
    validator = SQLSafetyValidator(default_limit=default_limit, max_limit=max_limit)
    return validator.validate(raw_sql)


def safe_execute_query(
    raw_sql: str,
    params: Sequence[Any] | None = None,
    default_limit: int = DEFAULT_ROW_LIMIT,
    max_limit: int = MAX_ROW_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    db_path: Path | str | None = None,
) -> ReadOnlyQueryResult:
    """
    Convenience function to validate, sanitize, and execute query through
    the read-only database connection layer.
    """
    validator = SQLSafetyValidator(
        default_limit=default_limit,
        max_limit=max_limit,
        db_path=db_path,
    )
    return validator.execute(raw_sql=raw_sql, params=params, timeout=timeout)
