"""
Gemini embedding adapter.

Translates our provider-agnostic EmbeddingParams into a real call against
Google's Gemini embedding API (`gemini-embedding-001`), and translates
the response back into our provider-agnostic EmbeddingResult. Same
`google-genai` SDK as `generate/backends/gemini.py` - one client
constructor pattern, two different methods on it.
"""

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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

# Generic (Section 3.5) -> Gemini's own task_type vocabulary. Anything not
# in this map (an unrecognized generic value, or task_type=None) is sent
# with no task_type hint at all - Gemini still embeds the text, just
# without the retrieval-quality boost a correct hint gives.
_TASK_TYPE_MAP = {
    "query": "RETRIEVAL_QUERY",
    "document": "RETRIEVAL_DOCUMENT",
    "similarity": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


class GeminiEmbeddingBackend(EmbeddingBackend):
    """Adapter for Gemini embedding (Google AI Studio free tier)."""

    name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-embedding-001") -> None:
        """
        Args:
            api_key: Google AI Studio API key (from env, never hardcoded).
            model_name: Which Gemini embedding model to call. Kept as a
                constructor arg, not hardcoded in `embed` - same
                free-tier-catalog-drift reasoning as the generation adapter.
        """
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def embed(self, params: EmbeddingParams) -> EmbeddingResult:
        if not params.texts:
            # Reject before ever reaching the provider - an empty batch
            # isn't a provider failure, it's a caller mistake the schema
            # layer should really have caught (see schemas/embed.py's
            # min_length constraint); this is a defensive second check.
            raise EmbeddingBackendError("embed() called with an empty texts list.")

        config = genai_types.EmbedContentConfig(
            task_type=_TASK_TYPE_MAP.get(params.task_type or "", None),
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=self.model_name,
                contents=params.texts,
                config=config,
            )
        except genai_errors.APIError as e:
            raise self._classify_error(e) from e

        vectors = [list(embedding.values) for embedding in response.embeddings]
        dimensions = len(vectors[0]) if vectors else 0

        return EmbeddingResult(
            vectors=vectors,
            model_name=self.model_name,
            dimensions=dimensions,
            # Gemini's embed_content response doesn't report token usage
            # the way generate_content's does - no usage_metadata to read.
            total_tokens=None,
        )

    async def health_check(self) -> HealthStatus:
        """Lightweight check per Section 3.8: confirms reachability + valid
        key without a full embedding call, by listing available models -
        identical approach to the generation adapter's health_check."""
        try:
            async for _ in await self._client.aio.models.list():
                break
            return HealthStatus(backend=self.name, reachable=True)
        except genai_errors.APIError as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))
        except Exception as e:  # network errors, etc. - still "not reachable"
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

    @staticmethod
    def _classify_error(e: genai_errors.APIError) -> EmbeddingBackendError:
        """Same classification approach as the generation adapter's
        `_classify_error` - see that file for the "verify against your
        installed SDK version" caveat, which applies here identically."""
        code = getattr(e, "code", None)
        message = str(e).lower()
        if code == 401 or code == 403:
            return BackendAuthError(str(e))
        if code == 429:
            if "quota" in message or "per day" in message:
                return QuotaExceededError(str(e))
            return RateLimitedError(str(e))
        return EmbeddingBackendError(str(e))
