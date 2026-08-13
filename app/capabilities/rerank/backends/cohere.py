"""
Cohere rerank adapter.

The only backend in scope for G4 (Section 3.3) - unlike generate/embed,
there's no free-tier fallback provider for reranking, so this is
deliberately a one-backend capability. See rerank/router.py's docstring
for how the shared resilience/fallback machinery behaves with a
one-item attempt order (it just works - no special-casing needed).

Uses the official `cohere` SDK's async v2 client, same as the embedding
adapter. Install with:
    pip install cohere
"""

from cohere import AsyncClientV2
from cohere.core.api_error import ApiError

from app.capabilities.rerank.base import (
    BackendAuthError,
    HealthStatus,
    RateLimitedError,
    RerankBackend,
    RerankBackendError,
    RerankedItem,
    RerankParams,
    RerankResult,
)


class CohereRerankBackend(RerankBackend):
    """Adapter for Cohere Rerank (free tier)."""

    name = "cohere"

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0") -> None:
        """
        Args:
            api_key: Cohere API key (from env, never hardcoded).
            model_name: Which Cohere rerank model to call. Kept as a
                constructor arg, not hardcoded in `rerank` - same
                free-tier-catalog-drift reasoning as every other adapter.
                Cohere's newer `rerank-v3.5` is multilingual and may be
                worth switching to later; kept at the English-only v3.0
                default for now to match this gateway's other
                English-first defaults (e.g. `embed-english-v3.0`).
        """
        self._client = AsyncClientV2(api_key=api_key)
        self.model_name = model_name

    async def rerank(self, params: RerankParams) -> RerankResult:
        if not params.documents:
            # Reject before ever reaching the provider - reordering an
            # empty list isn't a provider failure, it's a caller mistake
            # the schema layer should really have caught (see
            # schemas/rerank.py's min_length constraint); this is a
            # defensive second check, same pattern as the embed adapters.
            raise RerankBackendError("rerank() called with an empty documents list.")

        try:
            response = await self._client.rerank(
                model=self.model_name,
                query=params.query,
                documents=params.documents,
                top_n=params.top_n,
            )
        except ApiError as e:
            raise self._classify_error(e) from e

        # Cohere's response is ALREADY sorted most-to-least relevant, and
        # each result's `index` refers back to `params.documents`'
        # original position - we just look the text up by that index to
        # give callers the reordered text alongside its score, rather
        # than making every caller re-index into their own original list.
        results = [
            RerankedItem(
                index=item.index,
                text=params.documents[item.index],
                relevance_score=item.relevance_score,
            )
            for item in response.results
        ]

        return RerankResult(results=results, model_name=self.model_name)

    async def health_check(self) -> HealthStatus:
        """Lightweight check per Section 3.8: a minimal two-document
        rerank call. Like the Cohere embedding adapter, there's no
        cheaper "ping" endpoint on this SDK's client, so this costs one
        small real call - kept to two short documents to keep it as
        close to free as this provider allows."""
        try:
            await self._client.rerank(
                model=self.model_name,
                query="ping",
                documents=["a", "b"],
            )
            return HealthStatus(backend=self.name, reachable=True)
        except ApiError as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))
        except Exception as e:  # network errors, etc. - still "not reachable"
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

    @staticmethod
    def _classify_error(e: ApiError) -> RerankBackendError:
        """Same classification approach as the Cohere embedding adapter's
        `_classify_error` - see that file for the "verify against your
        installed SDK version" caveat, which applies here identically."""
        status_code = getattr(e, "status_code", None)
        if status_code in (401, 403):
            return BackendAuthError(str(e))
        if status_code == 429:
            return RateLimitedError(str(e))
        return RerankBackendError(str(e))
