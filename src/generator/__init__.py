"""
SQL Generator package for Role B Data Service.
"""

from .prompts import NORTHWIND_SCHEMA_PROMPT
from .sql_generator import (
    DEFAULT_GROQ_MODEL,
    GenerationResult,
    GroqConfigurationError,
    GroqProviderError,
    GroqSQLGenerator,
    SQLGenerationError,
    clean_markdown_fences,
)

__all__ = [
    "DEFAULT_GROQ_MODEL",
    "GenerationResult",
    "GroqConfigurationError",
    "GroqProviderError",
    "GroqSQLGenerator",
    "NORTHWIND_SCHEMA_PROMPT",
    "SQLGenerationError",
    "clean_markdown_fences",
]
