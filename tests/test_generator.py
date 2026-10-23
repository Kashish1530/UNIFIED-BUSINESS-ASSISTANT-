"""
Offline unit tests for GroqSQLGenerator.
Verifies SQL generation, markdown cleanup, model output safety gating,
empty/malformed response handling, configuration errors, and provider error wrapping
without making any network calls or requiring a real API key.
"""

from unittest.mock import MagicMock, patch
import unittest

import groq
from groq import APIConnectionError, AuthenticationError, RateLimitError

from src.database.connection import execute_read_only_query
from src.generator.sql_generator import (
    DEFAULT_GROQ_MODEL,
    GenerationResult,
    GroqConfigurationError,
    GroqProviderError,
    GroqSQLGenerator,
    SQLGenerationError,
    clean_markdown_fences,
)


def create_mock_completion(content: str):
    """Helper to build a mock Groq ChatCompletion object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


class TestGroqSQLGenerator(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.generator = GroqSQLGenerator(
            client=self.mock_client,
            model="llama-3.3-70b-versatile",
        )

    # -------------------------------------------------------------------------
    # 1. Configuration & Initialization Tests
    # -------------------------------------------------------------------------
    def test_01_missing_api_key_raises_configuration_error(self):
        """Verify initialization fails with GroqConfigurationError if API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(GroqConfigurationError) as ctx:
                GroqSQLGenerator(api_key=None, client=None)
            self.assertIn("GROQ_API_KEY", str(ctx.exception))

    def test_02_custom_and_default_model_configuration(self):
        """Verify model configuration honors defaults, arguments, and environment variables."""
        # 1. Default model
        g_default = GroqSQLGenerator(client=self.mock_client)
        self.assertEqual(g_default.model, DEFAULT_GROQ_MODEL)

        # 2. Argument override
        g_custom = GroqSQLGenerator(client=self.mock_client, model="llama-3.1-8b-instant")
        self.assertEqual(g_custom.model, "llama-3.1-8b-instant")

        # 3. Environment variable override
        with patch.dict("os.environ", {"GROQ_MODEL": "custom-groq-model"}):
            g_env = GroqSQLGenerator(client=self.mock_client)
            self.assertEqual(g_env.model, "custom-groq-model")

    def test_03_empty_question_raises_value_error(self):
        """Verify empty or whitespace-only questions raise ValueError."""
        with self.assertRaises(ValueError):
            self.generator.generate_sql("")
        with self.assertRaises(ValueError):
            self.generator.generate_sql("   \t\n ")

    # -------------------------------------------------------------------------
    # 2. Successful SQL Generation & Markdown Stripping
    # -------------------------------------------------------------------------
    def test_04_successful_clean_sql_generation(self):
        """Verify clean raw SQL output produces validated GenerationResult with default limit."""
        raw_sql = "SELECT CustomerID, CustomerName FROM Customers WHERE Country = 'Germany'"
        self.mock_client.chat.completions.create.return_value = create_mock_completion(raw_sql)

        result = self.generator.generate_sql("Which customers are in Germany?")
        self.assertIsInstance(result, GenerationResult)
        self.assertIn("LIMIT 100", result.sql)
        self.assertIn("Customers", result.tables_referenced)
        self.assertEqual(result.model, "llama-3.3-70b-versatile")
        self.assertEqual(result.raw_response, raw_sql)

    def test_05_markdown_fences_cleanup(self):
        """Verify markdown code blocks (```sql ... ```) are stripped before safety validation."""
        fenced_sql = "```sql\nSELECT ProductID, ProductName, Price FROM Products ORDER BY Price DESC;\n```"
        self.mock_client.chat.completions.create.return_value = create_mock_completion(fenced_sql)

        result = self.generator.generate_sql("Show most expensive products")
        self.assertNotIn("```", result.sql)
        self.assertIn("SELECT", result.sql)
        self.assertIn("LIMIT 100", result.sql)
        self.assertIn("Products", result.tables_referenced)

    def test_06_clean_markdown_fences_utility(self):
        """Verify clean_markdown_fences helper handles various formats."""
        self.assertEqual(
            clean_markdown_fences("```sql\nSELECT 1;\n```"),
            "SELECT 1;",
        )
        self.assertEqual(
            clean_markdown_fences("```\nSELECT 1;\n```"),
            "SELECT 1;",
        )
        self.assertEqual(
            clean_markdown_fences("  SELECT 1;  "),
            "SELECT 1;",
        )

    # -------------------------------------------------------------------------
    # 3. Untrusted Output Safety Gating (Destructive / Illegal SQL)
    # -------------------------------------------------------------------------
    def test_07_model_output_destructive_sql_blocked(self):
        """Verify generated DROP TABLE statements are blocked by safety validator."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion("DROP TABLE Customers;")

        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("Delete customers table")
        self.assertIn("safety validation", str(ctx.exception).lower())

    def test_08_model_output_multi_statement_blocked(self):
        """Verify multiple statements returned by model are rejected."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion(
            "SELECT * FROM Customers; DROP TABLE Orders;"
        )

        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("Get customers and orders")
        self.assertIn("multiple sql statements", str(ctx.exception).lower())

    def test_09_model_output_unknown_table_blocked(self):
        """Verify hallucinated non-existent tables are rejected."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion(
            "SELECT * FROM Users WHERE active = 1"
        )

        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("Show all active users")
        self.assertIn("unpermitted table", str(ctx.exception).lower())

    def test_10_model_output_unknown_column_blocked(self):
        """Verify hallucinated non-existent columns are rejected by compile check."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion(
            "SELECT NonExistentColumn FROM Customers"
        )

        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("Show non-existent data")
        self.assertIn("no such column", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 4. Malformed and Empty Responses
    # -------------------------------------------------------------------------
    def test_11_empty_response_raises_generation_error(self):
        """Verify empty responses from model raise SQLGenerationError."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion("")
        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("List all shippers")
        self.assertIn("empty response", str(ctx.exception).lower())

    def test_12_non_sql_response_raises_generation_error(self):
        """Verify conversational non-SQL text from model raises SQLGenerationError."""
        self.mock_client.chat.completions.create.return_value = create_mock_completion(
            "I apologize, but I do not have access to that information."
        )
        with self.assertRaises(SQLGenerationError) as ctx:
            self.generator.generate_sql("What is the weather today?")
        self.assertIn("syntax error", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 5. Groq Provider / API Error Handling
    # -------------------------------------------------------------------------
    def test_13_groq_api_connection_error_wrapped(self):
        """Verify APIConnectionError is caught and wrapped in GroqProviderError."""
        mock_request = MagicMock()
        self.mock_client.chat.completions.create.side_effect = APIConnectionError(
            request=mock_request,
            message="Connection refused",
        )

        with self.assertRaises(GroqProviderError) as ctx:
            self.generator.generate_sql("Get top products")
        self.assertIn("Connection refused", str(ctx.exception))

    def test_14_groq_rate_limit_error_wrapped(self):
        """Verify RateLimitError is caught and wrapped in GroqProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        self.mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )

        with self.assertRaises(GroqProviderError) as ctx:
            self.generator.generate_sql("Get top products")
        self.assertIn("Rate limit", str(ctx.exception))

    def test_15_groq_authentication_error_wrapped(self):
        """Verify AuthenticationError is caught and wrapped in GroqProviderError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        self.mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=mock_response,
            body={"error": {"message": "Invalid API Key"}},
        )

        with self.assertRaises(GroqProviderError) as ctx:
            self.generator.generate_sql("Get top products")
        self.assertIn("Invalid API Key", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 6. Model Verification Tests
    # -------------------------------------------------------------------------
    def test_16_verify_model_supported_true(self):
        """Verify verify_model_supported returns True when model is listed."""
        m1 = MagicMock()
        m1.id = "llama-3.3-70b-versatile"
        m2 = MagicMock()
        m2.id = "llama-3.1-8b-instant"
        mock_list = MagicMock()
        mock_list.data = [m1, m2]
        self.mock_client.models.list.return_value = mock_list

        self.assertTrue(self.generator.verify_model_supported())

    def test_17_verify_model_supported_false(self):
        """Verify verify_model_supported returns False when model is deprecated or missing."""
        m1 = MagicMock()
        m1.id = "other-model"
        mock_list = MagicMock()
        mock_list.data = [m1]
        self.mock_client.models.list.return_value = mock_list

        self.assertFalse(self.generator.verify_model_supported())

    def test_18_verify_model_supported_api_error_wrapped(self):
        """Verify verify_model_supported wraps API errors in GroqProviderError."""
        self.mock_client.models.list.side_effect = groq.GroqError("Network failure")
        with self.assertRaises(GroqProviderError):
            self.generator.verify_model_supported()

    # -------------------------------------------------------------------------
    # 7. End-to-End Pipeline Execution (Generation -> Safe Read-Only Execution)
    # -------------------------------------------------------------------------
    def test_19_end_to_end_generation_and_execution(self):
        """Verify generated and validated SQL executes successfully against northwind.db."""
        mock_sql = """
        SELECT c.CustomerName, COUNT(o.OrderID) AS OrderCount
        FROM Customers c
        JOIN Orders o ON c.CustomerID = o.CustomerID
        GROUP BY c.CustomerID, c.CustomerName
        ORDER BY OrderCount DESC
        LIMIT 5;
        """
        self.mock_client.chat.completions.create.return_value = create_mock_completion(mock_sql)

        gen_result = self.generator.generate_sql("Which 5 customers have the most orders?")
        self.assertEqual(gen_result.tables_referenced, ["Customers", "Orders"])

        # Execute through existing read-only layer
        query_result = execute_read_only_query(gen_result.sql)
        self.assertEqual(query_result.columns, ["CustomerName", "OrderCount"])
        self.assertEqual(query_result.row_count, 5)
        # Verify first row data (Customer Ernst Handel has 10 orders)
        first_row = query_result.rows[0]
        self.assertEqual(first_row["CustomerName"], "Ernst Handel")
        self.assertEqual(first_row["OrderCount"], 10)


if __name__ == "__main__":
    unittest.main()
