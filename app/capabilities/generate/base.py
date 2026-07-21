"""
Adapter interface for generation backends.

Every provider (Gemini, Groq, local Qwen via Ollama, ...) gets ONE adapter
class that implements `GenerationBackend`. Nothing outside `backends/`
should ever import a provider's SDK directly - the router and main.py only
ever talk to this interface.

Why a dataclass request/response instead of passing the FastAPI Pydantic
models straight through? Because the Pydantic request model is shaped by
the HTTP API (what a caller sends over the wire), while the backend only
needs the subset of fields relevant to generation. Keeping them separate
means changing the HTTP schema (e.g. adding a new optional API field)
doesn't force every adapter to change too.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# G3 refactor: the error taxonomy and HealthStatus used to be defined
# directly in this file. Embedding backends need the exact same
# distinctions (a Cohere 429 is still "retry vs failover" the same way a
# Gemini 429 is), so both now live in capabilities/common/ and this file
# just re-exports them - every existing
# "from app.capabilities.generate.base import BackendAuthError" (etc.)
# import elsewhere in the codebase keeps working unchanged, since these
# names now point at the same shared classes rather than a local copy.
from app.capabilities.common.errors import (  # noqa: F401
    BackendAuthError,
    BackendUnavailableError,
    GenerationBackendError,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.common.health import HealthStatus  # noqa: F401


@dataclass
class GenerationParams:
    """Everything an adapter needs to make one generation call.

    These fields are the real, provider-agnostic knobs every LLM API exposes in some form.
    """

    prompt: str
    system_instruction: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = field(default_factory=list)
    image_base64: str | None = None
    image_mime_type: str | None = None


@dataclass
class GenerationResult:
    """What an adapter hands back after a successful call.

    `raw_response` is intentionally NOT included here - 
    no-content-logging rule means we never want a code path where the
    full provider response accidentally ends up in a log row.
    """

    text: str
    model_name: str
    prompt_tokens: int | None
    completion_tokens: int | None


class GenerationBackend(ABC):
    """The contract every generation provider adapter must implement."""

    #: Short machine-readable name used in the registry, logs, and API
    #: requests (e.g. "gemini", "groq", "qwen_local"). Set by subclasses.
    name: str

    @abstractmethod
    async def generate(self, params: GenerationParams) -> GenerationResult:
        """Run one generation call against this backend.

        Must raise RateLimitedError / QuotaExceededError / BackendAuthError
        (not a bare Exception) so upstream resilience logic can react
        correctly to *why* the call failed.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Cheap reachability check.

        Must NOT be a full generation call. For hosted APIs this should be
        a minimal ping/models-list request; for local backends, a check
        that the process is up and the model is loaded.
        """
        raise NotImplementedError