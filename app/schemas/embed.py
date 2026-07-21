"""
HTTP-facing request/response models for POST /v1/embed.

Same separation as schemas/generate.py: this is "what the wire looks
like" (validated by Pydantic before any adapter runs, per Section 7.5's
"reject malformed requests early" principle), not "what an adapter needs"
(that's capabilities/embed/base.py's EmbeddingParams).
"""

from typing import Literal

from pydantic import BaseModel, Field

#: Mirrors capabilities/embed/base.py's TASK_TYPES - kept as an explicit
#: Literal here (rather than importing that tuple) so this file stays a
#: pure HTTP-schema file with no dependency on the adapter layer, same
#: as generate.py never importing from generate/base.py.
EmbedTaskType = Literal["query", "document", "similarity", "classification", "clustering"]


class EmbedRequest(BaseModel):
    """Body for POST /v1/embed."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch of texts to embed. Capped at 100 per request - "
        "callers embedding a larger corpus should send multiple requests "
        "rather than one unbounded batch (keeps a single request from "
        "tying up a backend connection indefinitely and bounds memory use "
        "server-side).",
    )
    task_type: EmbedTaskType | None = Field(
        default=None,
        description="Hint affecting embedding quality for retrieval: 'document' "
        "when embedding content to be searched, 'query' when embedding a "
        "search query. Applied where the backend supports it.",
    )
    backend: str | None = Field(
        default=None,
        description="Pin a specific backend by name (e.g. 'gemini'). "
        "Omit to use the configured primary for this capability.",
    )

    @property
    def has_blank_text(self) -> bool:
        """True if any entry in `texts` is empty/whitespace-only.

        Exposed as a property (checked explicitly in routes.py) rather
        than a `model_validator` that raises, so the 400 response can
        name exactly which behavior was rejected and why, consistent
        with how other validation errors in this gateway are surfaced.
        """
        return any(not t.strip() for t in self.texts)


class EmbedResponse(BaseModel):
    """The standardized envelope (Section 3.5), extended with the one
    field that's genuinely embedding-specific: `dimensions`. Every other
    field is identical in meaning to GenerateResponse's."""

    data: list[list[float]]
    dimensions: int
    backend_used: str
    model_name: str
    capability: Literal["embed"] = "embed"
    request_id: str
    latency_ms: int
    tokens_used: int | None
    cost_estimate: float
    retries: int = 0
