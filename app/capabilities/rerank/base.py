"""
Adapter interface for rerank backends.

Third capability, same five-file shape as generate/ and embed/ (Section
4). The output shape is the third genuinely different one Section 2
called out: generation returns text, embedding returns vectors, rerank
returns the SAME input list, reordered, each item scored by relevance to
a query. Not new content, not a new representation - a reordering.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# Same taxonomy every other capability uses - see common/errors.py's
# module docstring for why this lives outside generate/ now.
from app.capabilities.common.errors import (  # noqa: F401
    BackendAuthError,
    BackendUnavailableError,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.common.errors import BackendError as RerankBackendError  # noqa: F401
from app.capabilities.common.health import HealthStatus  # noqa: F401


@dataclass
class RerankParams:
    """Everything an adapter needs to make one rerank call."""

    query: str
    documents: list[str]
    #: Cap how many reordered results come back. None = return all of them,
    #: reordered. Matches Cohere's own `top_n` semantics directly, so the
    #: adapter can pass it straight through with no translation.
    top_n: int | None = None


@dataclass
class RerankedItem:
    """One result: which original document this is, and how relevant it
    was judged to be to the query.

    `index` is the position of this document in the ORIGINAL
    `RerankParams.documents` list, not its position in this result list -
    callers need this to map a result back to whatever they know about
    that document (e.g. a Policy RAG chunk's source document/section),
    since the adapter only ever saw raw text, never an external ID.
    """

    index: int
    text: str
    relevance_score: float


@dataclass
class RerankResult:
    """What an adapter hands back after a successful call.

    `results` is already sorted by `relevance_score`, descending - most
    relevant document first - matching how every rerank provider's API
    already returns it, so no adapter needs to re-sort anything.
    """

    results: list[RerankedItem]
    model_name: str


class RerankBackend(ABC):
    """The contract every rerank provider adapter must implement."""

    #: Short machine-readable name used in the registry, logs, and API
    #: requests (e.g. "cohere"). Set by subclasses.
    name: str

    @abstractmethod
    async def rerank(self, params: RerankParams) -> RerankResult:
        """Run one rerank call against this backend.

        Must raise RateLimitedError / QuotaExceededError / BackendAuthError
        / BackendUnavailableError (not a bare Exception) so the shared
        resilience layer can react correctly to *why* the call failed -
        identical contract to GenerationBackend.generate and
        EmbeddingBackend.embed.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Cheap reachability check - must NOT be a full rerank call."""
        raise NotImplementedError
