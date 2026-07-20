"""
HTTP-facing request/response models for POST /v1/extract.

Field is named `json_schema`, not `schema` - Pydantic's BaseModel already
has schema-related methods (`model_json_schema()`), and naming a field
`schema` invites exactly the kind of confusion that's cheap to avoid now.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExtractRequest(BaseModel):
    """Body for POST /v1/extract."""

    prompt: str = Field(..., min_length=1)
    json_schema: dict[str, Any] = Field(
        ..., description="A JSON Schema (draft 7+) the response must validate against."
    )
    system_instruction: str | None = None
    backend: str | None = Field(
        default=None,
        description="Pin a specific backend by name (e.g. 'gemini'). "
        "Omit to use the configured primary for this capability.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Defaults to 0.0, unlike /v1/generate's 0.2 default - "
        "structured extraction wants determinism, not creative variation.",
    )
    max_tokens: int = Field(default=1024, gt=0)
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="How many times to re-prompt the model if its output "
        "fails JSON parsing or schema validation, before giving up with a 422.",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes (no data: URI prefix). "
        "Requires image_mime_type to be set alongside it.",
    )
    image_mime_type: str | None = None

    @model_validator(mode="after")
    def _validate_image_pair(self) -> "ExtractRequest":
        if self.image_base64 and not self.image_mime_type:
            raise ValueError("image_mime_type is required when image_base64 is provided.")
        return self


class ExtractResponse(BaseModel):
    """The standardized envelope (Section 3.5) - same family as
    GenerateResponse, but `data` is a validated object, not free text."""

    data: dict[str, Any]
    backend_used: str
    model_name: str
    capability: Literal["extract"] = "extract"
    request_id: str
    latency_ms: int
    tokens_used: int | None
    cost_estimate: float
    retries: int = 0