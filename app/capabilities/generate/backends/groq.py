"""
Groq generation adapter.

Translates our provider-agnostic GenerationParams into a real call against
Groq's API (OpenAI-compatible chat completions shape), and translates the
response back into our provider-agnostic GenerationResult.

Uses the official `groq` SDK. Install with:
    pip install groq
"""

from groq import AsyncGroq
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from app.capabilities.generate.base import (
    BackendAuthError,
    GenerationBackend,
    GenerationBackendError,
    GenerationParams,
    GenerationResult,
    HealthStatus,
    QuotaExceededError,
    RateLimitedError,
)


class GroqGenerationBackend(GenerationBackend):
    """Adapter for Groq generation (free tier, OpenAI-compatible chat API)."""

    name = "groq"

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> None:
        """
        Args:
            api_key: Groq API key (from env, never hardcoded).
            model_name: Which Groq-hosted model to call. Kept as a
                constructor arg, not hardcoded in `generate` - same
                free-tier-catalog-drift reasoning as Gemini's adapter.
        """
        self._client = AsyncGroq(api_key=api_key)
        self.model_name = model_name

    async def generate(self, params: GenerationParams) -> GenerationResult:
        # Groq/OpenAI shape wants a flat messages list, not a separate
        # system_instruction config field like Gemini - so the system
        # prompt (if any) becomes the first message, not a config kwarg.
        messages: list[dict] = []
        if params.system_instruction:
            messages.append({"role": "system", "content": params.system_instruction})

        # NOTE: Groq's current chat.completions API is text-only - no
        # image content part here. If params.image_base64 is set and the
        # caller routed to "groq", that's exactly the "unsupported_input"
        # gap the project plan already flagged (Section 3.9) - not fixed
        # here, but don't silently ignore the image either.
        if params.image_base64:
            raise GenerationBackendError(
                "Groq backend does not support image input; route image "
                "requests to a vision-capable backend (e.g. gemini)."
            )

        messages.append({"role": "user", "content": params.prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                stop=params.stop_sequences or None,
                # NOTE: Groq's API has no top_k parameter (unlike Gemini) -
                # this is a real provider difference, not an oversight.
                # params.top_k is silently unused for this backend.
            )
        except APIStatusError as e:
            raise self._classify_error(e) from e

        choice = response.choices[0]
        usage = response.usage
        return GenerationResult(
            text=choice.message.content or "",
            model_name=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    async def health_check(self) -> HealthStatus:
        """Lightweight check per Section 3.8: confirms reachability + valid
        key without a full generation call, by listing available models."""
        try:
            await self._client.models.list()
            return HealthStatus(backend=self.name, reachable=True)
        except APIStatusError as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))
        except (APIConnectionError, APITimeoutError) as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

    @staticmethod
    def _classify_error(e: APIStatusError) -> GenerationBackendError:
        """Map Groq's exception classes onto our RPM/TPM vs RPD vs auth
        taxonomy.

        Groq gives us distinct exception classes (cleaner than Gemini's
        single APIError+code), so auth is unambiguous. The RPM/TPM vs RPD
        split, though, isn't in the exception class at all - Groq signals
        it via response headers (`x-ratelimit-*` / `retry-after`), not a
        different error type. We don't have those headers wired through
        yet (that's G2's resilience-layer job per Section 3.6) - for now,
        every RateLimitError is treated as RPM/TPM (retryable), which is
        the safer default (worst case: G2 retries a few extra times before
        the header check is added, rather than skipping a real failover).
        Revisit this classification once G2 reads response headers.
        """
        if isinstance(e, AuthenticationError | PermissionDeniedError):
            return BackendAuthError(str(e))
        if isinstance(e, RateLimitError):
            return RateLimitedError(str(e))
        return GenerationBackendError(str(e))