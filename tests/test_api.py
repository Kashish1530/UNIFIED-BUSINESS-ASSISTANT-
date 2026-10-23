
"""Offline contract tests for the Role B mock API."""

import unittest

from fastapi.testclient import TestClient

from main import app


class RoleBAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "role-b-data-service"},
        )

    def test_valid_query_returns_expected_contract(self):
        response = self.client.post(
            "/query",
            json={"question": "Show top products"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertTrue(body["success"])
        self.assertIsNotNone(body["sql"])
        self.assertEqual(body["columns"], ["ProductName", "TotalSales"])
        self.assertEqual(len(body["rows"]), body["row_count"])
        self.assertEqual(body["row_count"], 2)
        self.assertIsNone(body["error"])
        self.assertIn("MOCK ONLY", body["sql"])

    def test_whitespace_question_returns_structured_failure(self):
        response = self.client.post(
            "/query",
            json={"question": "   "},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "success": False,
                "sql": None,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": "Question must not be empty or whitespace.",
            },
        )

    def test_empty_question_is_rejected_by_request_validation(self):
        response = self.client.post(
            "/query",
            json={"question": ""},
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_question_is_rejected_by_request_validation(self):
        response = self.client.post("/query", json={})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
