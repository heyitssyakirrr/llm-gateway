"""
HTTP-facing request/response models.

These are deliberately separate from `backends/base.py`'s dataclasses:
this file is "what the wire looks like", that file is "what an adapter
needs". Pydantic validates every request here BEFORE it reaches any
adapter (Section 7.5's Security checklist - reject malformed requests
early).
"""

from typing import Literal

from pydantic import BaseModel, Field


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


class BackendHealth(BaseModel):
    backend: str
    reachable: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """GET /v1/health - per-backend status, never one aggregate boolean
    (Section 3.8)."""

    backends: list[BackendHealth]
