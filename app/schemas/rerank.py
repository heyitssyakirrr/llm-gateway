"""
HTTP-facing request/response models for POST /v1/rerank.

Same separation as schemas/generate.py and schemas/embed.py: this is
"what the wire looks like" (validated by Pydantic before any adapter
runs, per Section 7.5's "reject malformed requests early" principle),
not "what an adapter needs" (that's capabilities/rerank/base.py's
RerankParams).
"""

from typing import Literal

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Body for POST /v1/rerank."""

    query: str = Field(..., min_length=1, description="The search query to rank documents against.")
    documents: list[str] = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Candidate texts to reorder by relevance to `query`. "
        "Capped at 1000 per request, matching Cohere's own recommended "
        "limit - callers with a larger candidate set should narrow it "
        "(e.g. via embedding-similarity search) before reranking, not "
        "send everything through reranking directly. Requires at least "
        "2 documents - reranking a single document has nothing to "
        "reorder against.",
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description="Return only the top N reordered results. Omit to "
        "get every input document back, reordered.",
    )
    backend: str | None = Field(
        default=None,
        description="Pin a specific backend by name (e.g. 'cohere'). "
        "Omit to use the configured primary for this capability.",
    )

    @property
    def has_blank_document(self) -> bool:
        """True if any entry in `documents` is empty/whitespace-only.

        Exposed as a property (checked explicitly in routes.py), same
        pattern as EmbedRequest.has_blank_text - lets the 400 response
        name exactly which behavior was rejected and why.
        """
        return any(not d.strip() for d in self.documents)


class RerankedItemResponse(BaseModel):
    """One reordered result."""

    index: int = Field(..., description="Position of this document in the ORIGINAL `documents` list sent in the request.")
    text: str
    relevance_score: float


class RerankResponse(BaseModel):
    """The standardized envelope (Section 3.5), extended with the one
    field that's genuinely rerank-specific: `data` here is
    RerankedItemResponse objects (index + text + score), not raw text or
    vectors - the reordering itself IS the output."""

    data: list[RerankedItemResponse]
    backend_used: str
    model_name: str
    capability: Literal["rerank"] = "rerank"
    request_id: str
    latency_ms: int
    # Cohere bills rerank by "search units," not input/output tokens, so
    # this stays None for rerank rather than misusing a tokens field for
    # a different unit - see routes.py's params_used logging instead for
    # document-count metadata.
    tokens_used: int | None = None
    cost_estimate: float
    retries: int = 0
