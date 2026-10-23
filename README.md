
# Unified Business Assistant â€” Role B

Role B provides the database and data-query service for the Unified Business Assistant project.

## Current status

The `/query` endpoint is currently a **mock API** for Role C integration. It does not generate SQL, call Groq, or execute database queries. Successful requests return fixed sample data.

The Northwind SQLite database and schema documentation are maintained separately.

## Requirements

- Python
- Dependencies listed in `requirements.txt`

## Setup

Activate your virtual environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the API

From the project root:

```powershell
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000` by default.

Interactive API documentation:
`http://127.0.0.1:8000/docs`

## Endpoints

### GET `/health`

Example response:

```json
{
  "status": "ok",
  "service": "role-b-data-service"
}
```

### POST `/query`

Request:

```json
{
  "question": "Show the top 5 products by sales"
}
```

Example mock response:

```json
{
  "success": true,
  "sql": "-- MOCK ONLY: SQL was not generated or executed",
  "columns": ["ProductName", "TotalSales"],
  "rows": [
    ["Sample Product A", 1500.0],
    ["Sample Product B", 1200.0]
  ],
  "row_count": 2,
  "error": null
}
```

Response fields:

| Field | Type | Meaning |
|---|---|---|
| `success` | boolean | Whether the mock request succeeded |
| `sql` | string or null | Mock marker; not executed SQL |
| `columns` | string array | Result column names |
| `rows` | array | Sample result rows |
| `row_count` | integer | Number of returned rows |
| `error` | string or null | Error message, if applicable |

A missing or empty `question` is rejected by request validation with HTTP 422. A whitespace-only question returns `success: false` with an error message.

## Run tests

Run the API tests:

```powershell
python -m unittest discover -s tests -p 'test_api.py' -v
```

Run the complete test suite:

```powershell
python -m unittest discover -s tests -v
```

## Database documentation

See [`docs/northwind_schema.md`](docs/northwind_schema.md) for the database schema and relationships.

## Integration note

The /query endpoint is currently a mock and does not generate or execute SQL. Agree on the response contract with Role C before replacing the mock implementation. Sample rows are demonstration data, not actual database results.
