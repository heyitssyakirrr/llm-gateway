"""
Gemini generation adapter.

Translates our provider-agnostic GenerationParams into a real call against
Google's Gemini Developer API, and translates the response back into our
provider-agnostic GenerationResult.

Uses the `google-genai` SDK (the current unified SDK - the older
`google-generativeai` package is deprecated). Install with:
    pip install google-genai
"""

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.backends.base import (
    BackendAuthError,
    GenerationBackend,
    GenerationBackendError,
    GenerationParams,
    GenerationResult,
    HealthStatus,
    QuotaExceededError,
    RateLimitedError,
)


class GeminiGenerationBackend(GenerationBackend):
    """Adapter for Gemini generation (Google AI Studio free tier)."""

    name = "gemini"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        """
        Args:
            api_key: Google AI Studio API key (from env, never hardcoded).
            model_name: Which Gemini model to call. Kept as a constructor
                arg (not hardcoded in `generate`) so config.py controls it -
                free-tier model catalogs change without notice (Section 7).
        """
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def generate(self, params: GenerationParams) -> GenerationResult:
        config = genai_types.GenerateContentConfig(
            system_instruction=params.system_instruction,
            temperature=params.temperature,
            max_output_tokens=params.max_tokens,
            top_p=params.top_p,
            top_k=params.top_k,
            stop_sequences=params.stop_sequences or None,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=params.prompt,
                config=config,
            )
        except genai_errors.APIError as e:
            raise self._classify_error(e) from e

        usage = response.usage_metadata
        return GenerationResult(
            text=response.text or "",
            model_name=self.model_name,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
        )

    async def health_check(self) -> HealthStatus:
        """Lightweight check per Section 3.8: confirms reachability + valid key
        without a full generation call, by listing available models."""
        try:
            # aio.models.list() is a paginator; we only need to know it doesn't
            # raise. Pulling one page is enough to prove the key/network work.
            async for _ in await self._client.aio.models.list():
                break
            return HealthStatus(backend=self.name, reachable=True)
        except genai_errors.APIError as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))
        except Exception as e:  # network errors, etc. - still "not reachable"
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

    @staticmethod
    def _classify_error(e: genai_errors.APIError) -> GenerationBackendError:
        """Map Gemini's error codes onto our RPM/TPM vs RPD vs auth taxonomy.

        NOTE: verify these codes against the installed SDK version when you
        run this for real - provider error shapes drift. code 429 covers
        both rolling-window rate limits AND quota exhaustion for Gemini;
        the response body's `status`/`message` usually distinguishes them
        (e.g. "RESOURCE_EXHAUSTED" with a quota-related message vs a plain
        rate-limit message). Treat this as a first pass to refine once you
        see a real 429 in your own logs.
        """
        code = getattr(e, "code", None)
        message = str(e).lower()
        if code == 401 or code == 403:
            return BackendAuthError(str(e))
        if code == 429:
            if "quota" in message or "per day" in message:
                return QuotaExceededError(str(e))
            return RateLimitedError(str(e))
        return GenerationBackendError(str(e))
