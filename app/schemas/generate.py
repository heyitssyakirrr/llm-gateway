"""
HTTP-facing request/response models.

These are deliberately separate from `backends/base.py`'s dataclasses:
this file is "what the wire looks like", that file is "what an adapter
needs". Pydantic validates every request here BEFORE it reaches any
adapter (Section 7.5's Security checklist - reject malformed requests
early).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):
    """Body for POST /v1/generate."""

    prompt: str = Field(..., min_length=1)
    system_instruction: str | None = None
    backend: str | None = Field(
        default=None,
        description="Pin a specific backend by name (e.g. 'gemini'). "
        "Omit to use the configured primary for this capability.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    stop_sequences: list[str] = Field(default_factory=list)
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes (no data: URI prefix). "
        "Requires image_mime_type to be set alongside it.",
    )
    image_mime_type: str | None = Field(
        default=None,
        description="e.g. 'image/jpeg', 'image/png'. Required if image_base64 is set.",
    )

    @model_validator(mode="after")
    def _validate_image_pair(self) -> "GenerateRequest":
        if self.image_base64 and not self.image_mime_type:
            raise ValueError("image_mime_type is required when image_base64 is provided.")
        return self


class GenerateResponse(BaseModel):
    """The standardized envelope (Section 3.5) - identical shape regardless
    of which backend actually served the request."""

    data: str
    backend_used: str
    model_name: str
    capability: Literal["generate"] = "generate"
    request_id: str
    latency_ms: int
    tokens_used: int | None
    cost_estimate: float
    retries: int = 0