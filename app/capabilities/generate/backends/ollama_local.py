"""
Ollama (local Qwen) generation adapter.

Unlike gemini.py / groq.py, this backend has no API key, no rate limit,
and no daily quota - it's a process running on the same machine. That
changes what "can go wrong" actually looks like, on purpose (this is the
backend chosen in G1 specifically to stress-test whether the
GenerationBackend/GenerationParams interface generalizes beyond hosted
APIs - see the project plan's G1 section).

Uses the official `ollama` Python package. Install with:
    pip install ollama
"""

from ollama import AsyncClient
from ollama import RequestError, ResponseError

from app.capabilities.generate.base import (
    GenerationBackend,
    GenerationBackendError,
    GenerationParams,
    GenerationResult,
    HealthStatus,
)


class OllamaGenerationBackend(GenerationBackend):
    """Adapter for a locally-running Qwen model served via Ollama.

    Note on the error taxonomy: RateLimitedError / QuotaExceededError /
    BackendAuthError (defined in base.py) all describe failure modes that
    assume a remote, metered, authenticated API. None of them apply here
    - there's no key to reject, no per-minute window, no daily cap. The
    real local failure modes are different in kind, not just degree:
      - Ollama service isn't running at all (connection refused)
      - The model name isn't pulled yet (404-shaped ResponseError)
      - The machine is out of VRAM/RAM for this model
    None of these are "retry with backoff" or "fail over to another
    backend" situations - they're setup/environment problems. Rather than
    force-fit them into the hosted-API taxonomy, they're raised as the
    base GenerationBackendError with a clear message. This is exactly the
    kind of interface gap G1 is meant to surface - flagging it here for
    G2's resilience layer to make a deliberate decision about (e.g. a new
    BackendUnavailableError category), instead of quietly picking one.
    """

    name = "qwen_local"

    def __init__(self, host: str = "http://localhost:11434", model_name: str = "qwen2.5:3b-instruct") -> None:
        """
        Args:
            host: Ollama server address. Defaults to the standard local
                install - overridable via env var for e.g. a remote GPU box.
            model_name: Which pulled Ollama model to call. Same reasoning
                as Gemini/Groq's model_name arg: swapping models (e.g. to
                phi3.5:3.8b) is a config change, not a code change.
        """
        self._client = AsyncClient(host=host)
        self.model_name = model_name

    async def generate(self, params: GenerationParams) -> GenerationResult:
        messages: list[dict] = []
        if params.system_instruction:
            messages.append({"role": "system", "content": params.system_instruction})

        # qwen2.5:3b-instruct is text-only. Same "reject clearly" choice
        # as groq.py rather than silently dropping the image or letting
        # Ollama error out in a less legible way.
        if params.image_base64:
            raise GenerationBackendError(
                "Configured qwen_local model does not support image input; "
                "route image requests to a vision-capable backend (e.g. gemini)."
            )

        messages.append({"role": "user", "content": params.prompt})

        options = {
            "temperature": params.temperature,
            "num_predict": params.max_tokens,  # Ollama's name for max_tokens
            "top_p": params.top_p,
            "top_k": params.top_k,
            "stop": params.stop_sequences or None,
        }

        try:
            response = await self._client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )
        except ResponseError as e:
            # e.g. "model 'qwen2.5:3b-instruct' not found, try pulling it first"
            raise GenerationBackendError(
                f"Ollama rejected the request (status {e.status_code}): {e.error}"
            ) from e
        except RequestError as e:
            # Malformed request to the ollama client itself - a bug in how
            # we're calling it, not a runtime backend failure.
            raise GenerationBackendError(f"Invalid request to Ollama: {e}") from e
        except ConnectionError as e:
            # Ollama service isn't running / unreachable at `host`.
            raise GenerationBackendError(
                f"Could not reach Ollama at the configured host: {e}"
            ) from e

        return GenerationResult(
            text=response.message.content or "",
            model_name=response.model or self.model_name,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
        )

    async def health_check(self) -> HealthStatus:
        """Checks two things a hosted-API health check doesn't need to:
        1. Is the Ollama service reachable at all.
        2. Is *this specific model* actually pulled and available.

        A hosted backend's health check only needs to answer "is the API
        up" - the model always exists on the provider's side. Locally, the
        model itself is a thing that can simply be missing (never pulled,
        or a typo in OLLAMA_MODEL_NAME), which is a distinct failure worth
        surfacing separately.
        """
        try:
            models_response = await self._client.list()
        except ConnectionError as e:
            return HealthStatus(
                backend=self.name, reachable=False, detail=f"Ollama not reachable: {e}"
            )
        except (ResponseError, RequestError) as e:
            return HealthStatus(backend=self.name, reachable=False, detail=str(e))

        available_names = {m.model for m in models_response.models}
        if self.model_name not in available_names:
            return HealthStatus(
                backend=self.name,
                reachable=False,
                detail=(
                    f"Ollama is running, but model '{self.model_name}' is not pulled. "
                    f"Run: ollama pull {self.model_name}"
                ),
            )
        return HealthStatus(backend=self.name, reachable=True)