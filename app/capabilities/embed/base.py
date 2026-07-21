"""
Adapter interface for embedding backends.

Mirrors `capabilities/generate/base.py`'s shape exactly (Section 4 of the
project plan: every capability folder has the same five files). The only
real difference is what a call takes in and hands back: generation takes
one prompt and returns text; embedding takes a BATCH of texts and returns
one vector per text - "different shapes of problem", per the project
plan's Section 2, which is exactly why this is its own capability
interface instead of a bolt-on to GenerationBackend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Same taxonomy generation uses - see capabilities/common/errors.py's
# module docstring for why this lives outside generate/ now.
from app.capabilities.common.errors import (  # noqa: F401
    BackendAuthError,
    BackendUnavailableError,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.common.errors import BackendError as EmbeddingBackendError  # noqa: F401
from app.capabilities.common.health import HealthStatus  # noqa: F401

#: Generic, provider-agnostic task-type vocabulary exposed over the wire
#: (Section 3.5). Gemini and Cohere each use their own vocabulary
#: ("RETRIEVAL_DOCUMENT" vs "search_document") - translating a generic
#: value into a provider-specific one is each adapter's job, not the
#: caller's. "query" and "document" are the two that matter for RAG
#: retrieval quality; the rest are supported where the backend has an
#: equivalent.
TASK_TYPES = ("query", "document", "similarity", "classification", "clustering")


@dataclass
class EmbeddingParams:
    """Everything an adapter needs to make one embedding call.

    `texts` is a batch (list), not a single string, from day one - the
    Policy RAG project's ingestion pipeline needs to embed many chunks
    without one HTTP round-trip per chunk (Section 3.5).
    """

    texts: list[str]
    task_type: str | None = None


@dataclass
class EmbeddingResult:
    """What an adapter hands back after a successful call.

    One vector per input text, in the SAME order as `EmbeddingParams.texts`
    - callers rely on positional correspondence rather than the adapter
    echoing texts back, so order must never be reshuffled by an adapter.

    `total_tokens` is `None` where a provider's embedding endpoint doesn't
    report token usage at all (this varies by provider, unlike
    generation) - `pricing.py`'s `estimate_cost` already treats `None` as
    "unknown," so this doesn't need special-casing at the call site.
    """

    vectors: list[list[float]]
    model_name: str
    dimensions: int
    total_tokens: int | None = field(default=None)


class EmbeddingBackend(ABC):
    """The contract every embedding provider adapter must implement."""

    #: Short machine-readable name used in the registry, logs, and API
    #: requests (e.g. "gemini", "cohere"). Set by subclasses.
    name: str

    @abstractmethod
    async def embed(self, params: EmbeddingParams) -> EmbeddingResult:
        """Run one (possibly batched) embedding call against this backend.

        Must raise RateLimitedError / QuotaExceededError / BackendAuthError
        / BackendUnavailableError (not a bare Exception) so the shared
        resilience layer can react correctly to *why* the call failed -
        identical contract to `GenerationBackend.generate`.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Cheap reachability check - must NOT be a full embedding call."""
        raise NotImplementedError
