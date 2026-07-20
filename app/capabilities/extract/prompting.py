"""
Builds the system instruction that tells a model to emit schema-matching
JSON. This is deliberately a plain-text instruction (not a provider's
native "JSON mode" feature) so it works identically across every backend
- Gemini, Groq, and Qwen-local have three different (or absent) native
structured-output features, but every one of them can follow a clear
text instruction. Consistency across backends matters more here than
using each provider's fanciest option.
"""

import json
from typing import Any


def build_extraction_instruction(json_schema: dict[str, Any]) -> str:
    return (
        "You must respond with ONLY valid JSON that satisfies the JSON "
        "Schema below. Do not include markdown code fences, explanations, "
        "or any text before or after the JSON object.\n\n"
        f"JSON Schema:\n{json.dumps(json_schema, indent=2)}"
    )