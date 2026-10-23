"""
Groq-backed SQL Generation Module for Role B Data Service.
Generates safe, bounded SQLite queries from natural language business questions.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

import groq
from groq import GroqError

from src.database.connection import get_project_root
from src.generator.prompts import NORTHWIND_SCHEMA_PROMPT
from src.safety.validator import (
    DEFAULT_ROW_LIMIT,
    MAX_ROW_LIMIT,
    SQLSafetyValidator,
    SQLValidationError,
)

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class SQLGenerationError(Exception):
    """Base exception for SQL generation failures."""
    pass


class GroqConfigurationError(SQLGenerationError):
    """Raised when required Groq credentials or configuration are missing."""
    pass


class GroqProviderError(SQLGenerationError):
    """Raised when Groq API communication, authentication, or rate limits fail."""
    pass


@dataclass(frozen=True)
class GenerationResult:
    """Encapsulates generated and validated SQL query results."""
    question: str
    raw_response: str
    sql: str
    model: str
    tables_referenced: list[str]


def load_env_file(dotenv_path: Path | None = None):
    """Load key-value pairs from .env into os.environ if not already present."""
    if dotenv_path is None:
        dotenv_path = get_project_root() / ".env"
    if not dotenv_path.exists():
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def clean_markdown_fences(raw_text: str) -> str:
    """
    Strip markdown code fences (e.g. ```sql ... ```) and leading/trailing
    whitespace or quotation marks produced by LLM responses.
    """
    text = raw_text.strip()

    # Extract content inside markdown code blocks ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Remove single-line fence if improperly closed
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


class GroqSQLGenerator:
    """
    Generates verified read-only SQL queries from natural language questions
    using the Groq API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        validator: SQLSafetyValidator | None = None,
        default_limit: int = DEFAULT_ROW_LIMIT,
        max_limit: int = MAX_ROW_LIMIT,
    ):
        load_env_file()

        # Resolve API key (argument takes precedence over environment variable)
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")

        # Resolve model (argument > environment variable > default)
        self.model = model or os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL

        # Configure validator
        self.validator = validator or SQLSafetyValidator(
            default_limit=default_limit,
            max_limit=max_limit,
        )

        # Configure Groq client
        if client is not None:
            self.client = client
        else:
            if not self._api_key or not self._api_key.strip():
                raise GroqConfigurationError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please set GROQ_API_KEY or pass api_key to initialize the generator."
                )
            self.client = groq.Groq(api_key=self._api_key)

    def verify_model_supported(self) -> bool:
        """
        Verify that the configured model is currently available on Groq.
        Returns True if supported, False if not found.
        Raises GroqProviderError on authentication or connectivity issues.
        """
        try:
            models_response = self.client.models.list()
            available_ids = {m.id for m in models_response.data}
            return self.model in available_ids
        except GroqError as e:
            raise GroqProviderError(f"Failed to query Groq model list: {e}")
        except Exception as e:
            raise GroqProviderError(f"Unexpected error querying Groq models: {e}")

    def generate_sql(self, question: str) -> GenerationResult:
        """
        Generate, sanitize, and validate an SQL query from a natural language question.
        Guarantees:
        - Generated SQL passes SQLSafetyValidator before returning.
        - Untrusted output is cleaned of markdown formatting.
        - Raises SQLGenerationError if output fails safety, syntax, or schema rules.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NORTHWIND_SCHEMA_PROMPT},
                    {"role": "user", "content": question.strip()},
                ],
                temperature=0.0,
            )
        except GroqError as e:
            raise GroqProviderError(f"Groq API call failed: {e}")
        except Exception as e:
            raise GroqProviderError(f"Unexpected error communicating with Groq: {e}")

        # Extract text from response
        try:
            raw_text = chat_completion.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            raise SQLGenerationError(f"Malformed response structure from Groq: {e}")

        if not raw_text.strip():
            raise SQLGenerationError("Model returned an empty response.")

        clean_sql = clean_markdown_fences(raw_text)
        if not clean_sql:
            raise SQLGenerationError("Extracted query from model response is empty.")

        # Treat model SQL as untrusted: pass through safety validator
        val_result = self.validator.validate(clean_sql)
        if not val_result.is_valid:
            raise SQLGenerationError(
                f"Generated SQL failed safety validation: {val_result.error}\nGenerated SQL: {clean_sql}"
            )

        return GenerationResult(
            question=question,
            raw_response=raw_text,
            sql=val_result.sanitized_sql,
            model=self.model,
            tables_referenced=val_result.tables,
        )
