"""
Cohere embedding adapter.

Translates our provider-agnostic EmbeddingParams into a real call against
Cohere's Embed v3 API, and translates the response back into our
provider-agnostic EmbeddingResult.

Uses the official `cohere` SDK's async v2 client. Install with:
    pip install cohere
"""

from cohere import AsyncClientV2
from cohere.core.api_error import ApiError

from app.capabilities.embed.base import (
    BackendAuthError,
    EmbeddingBackend,
    EmbeddingBackendError,
    EmbeddingParams,
    EmbeddingResult,
    HealthStatus,
    QuotaExceededError,
    RateLimitedError,
)

# Generic (Section 3.5) -> Cohere's own input_type vocabulary. Unlike
# Gemini, Cohere's `embed()` REQUIRES an input_type on every call (there's
# no "no hint" option) - so anything not in this map, or task_type=None,
# falls back to "search_document". That's a deliberate default, not a
# guess: it's the safer of the two retrieval-relevant options, since
# document-embedding is what happens far more often (ingesting a corpus)
# than query-embedding (one query at a time) in a typical RAG pipeline.
_TASK_TYPE_MAP = {
    "query": "search_query",
    "document": "search_document",
    "classification": "classification",
    "clustering": "clustering",
    # Cohere has no direct "similarity" input_type - document embeddings
    # are Cohere's own recommended default for general similarity use.
    "similarity": "search_document",
}
_DEFAULT_INPUT_TYPE = "search_document"


class CohereEmbeddingBackend(EmbeddingBackend):
    """Adapter for Cohere embedding (free tier, Embed v3)."""

    name = "cohere"

    def __init__(self, api_key: str, model_name: str = "embed-english-v3.0") -> None:
        """
        Args:
            api_key: Cohere API key (from env, never hardcoded).
            model_name: Which Cohere embedding model to call. Kept as a
                constructor arg, not hardcoded in `embed` - same
                free-tier-catalog-drift reasoning as every other adapter.
        """
        self._client = AsyncClientV2(api_key=api_key)
        self.model_name = model_name

    async def embed(self, params: EmbeddingParams) -> EmbeddingResult:
        if not params.texts:
            raise EmbeddingBackendError("embed() called with an empty texts list.")

        input_type = _TASK_TYPE_MAP.get(params.task_type or "", _DEFAULT_INPUT_TYPE)

        try:
            response = await self._client.embed(
                model=self.model_name,
                input_type=input_type,
                texts=params.texts,
                embedding_types=["float"],
            )
        except ApiError as e:
            raise self._classify_error(e) from e

        vectors = list(response.embeddings.float or [])
        dimensions = len(vectors[0]) if vectors else 0
        total_tokens = None
        if response.meta and response.meta.billed_units and response.meta.billed_units.input_tokens is not None:
            total_tokens = int(response.meta.billed_units.input_tokens)

        return EmbeddingResult(
            vectors=vectors,
            model_name=self.model_name,
            dimensions=dimensions,
            total_tokens=total_tokens,
        )

    async def health_check(self) -> HealthStatus:
        """Lightweight check per Section 3.8: a single-word, single-text
        embed call. Cohere's SDK has no cheaper "list models" / "ping"
        endpoint on the v2 client, so unlike Gemini/Groq this check does
        cost one real (tiny) embedding call - kept to one word to keep it
        as close to free as this provider allows."""
        try:
            await self._client.embed(
                model=self.model_name,
                input_type=_DEFAULT_INPUT_TYPE,
                texts=["ping"],
                embedding_types=["float"],
            )
            return HealthStatus(backend=self.name, reachable=True)
        except ApiError as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))
        except Exception as e:  # network errors, etc. - still "not reachable"
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

    @staticmethod
    def _classify_error(e: ApiError) -> EmbeddingBackendError:
        """Map Cohere's status codes onto our RPM/TPM vs RPD vs auth
        taxonomy.

        NOTE: like the Gemini/Groq adapters' equivalents, verify these
        codes against the installed SDK/API version when you run this for
        real. Cohere doesn't cleanly separate RPM/TPM from RPD in the
        status code alone (both surface as 429) - treated here as
        RPM/TPM (retryable) by default, the same safer-default choice
        `groq.py` already documents for the identical ambiguity.
        """
        status_code = getattr(e, "status_code", None)
        if status_code in (401, 403):
            return BackendAuthError(str(e))
        if status_code == 429:
            return RateLimitedError(str(e))
        return EmbeddingBackendError(str(e))
