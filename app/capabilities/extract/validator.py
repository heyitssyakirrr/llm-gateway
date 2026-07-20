"""
Parses and validates a backend's raw text output against the caller's
JSON Schema.

Kept separate from routes.py because this logic has nothing to do with
HTTP or any particular backend - it's pure "does this text satisfy this
schema" logic, testable on its own with no FastAPI app or adapter involved.
"""

import json
from typing import Any

from jsonschema import ValidationError, validate


class ExtractionParseError(Exception):
    """Raised when a backend's output isn't valid JSON, or is valid JSON
    that doesn't satisfy the caller's schema.

    The message is written to be re-fed to the model on retry (see
    routes.py), so it deliberately describes *what* was wrong in plain
    language, not a Python traceback.
    """


def parse_and_validate(raw_text: str, json_schema: dict[str, Any]) -> dict[str, Any]:
    """Strip common formatting noise, parse as JSON, validate against schema.

    Raises ExtractionParseError (never a bare json/jsonschema exception)
    so callers only ever need to catch one thing.
    """
    text = _strip_code_fences(raw_text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExtractionParseError(f"Response was not valid JSON: {e}") from e

    try:
        validate(instance=parsed, schema=json_schema)
    except ValidationError as e:
        # e.json_path (e.g. "$.amount") pinpoints exactly which field
        # failed - much more useful to feed back to a model on retry than
        # the full jsonschema exception repr.
        raise ExtractionParseError(
            f"JSON did not match the required schema at '{e.json_path}': {e.message}"
        ) from e

    return parsed


def _strip_code_fences(text: str) -> str:
    """Models very commonly wrap JSON in ```json ... ``` even when told
    not to. Stripping this here, once, is cheaper and more reliable than
    hoping every prompt/every backend honors "no markdown" perfectly."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    lines = text.split("\n")
    lines = lines[1:]  # drop the opening ``` or ```json line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()