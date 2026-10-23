from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Role B - Data Service",
    description="Mock query API for Role C integration",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language business question",
        examples=["Show the top 5 products by sales"],
    )


class QueryResponse(BaseModel):
    success: bool
    sql: str | None
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    error: str | None


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok", "service": "role-b-data-service"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Return deterministic sample data without executing a query."""
    question = request.question.strip()

    if not question:
        return QueryResponse(
            success=False,
            sql=None,
            columns=[],
            rows=[],
            row_count=0,
            error="Question must not be empty or whitespace.",
        )

    # Deliberately fixed demonstration data.
    # These rows are not database results.
    columns = ["ProductName", "TotalSales"]
    rows: list[list[Any]] = [
        ["Sample Product A", 1500.00],
        ["Sample Product B", 1200.00],
    ]

    return QueryResponse(
        success=True,
        sql="-- MOCK ONLY: SQL was not generated or executed",
        columns=columns,
        rows=rows,
        row_count=len(rows),
        error=None,
    )
