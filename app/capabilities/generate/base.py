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


@dataclass
class GenerationParams:
    """Everything an adapter needs to make one generation call.

    This mirrors Section 7.6 of the project plan: these fields are the
    real, provider-agnostic knobs every LLM API exposes in some form.
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

    `raw_response` is intentionally NOT included here - Section 7's
    no-content-logging rule means we never want a code path where the
    full provider response accidentally ends up in a log row. If you need
    provider-specific debug info later, add a narrow, explicit field for
    it - don't smuggle the whole object through.
    """

    text: str
    model_name: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class HealthStatus:
    """Result of a single backend's /v1/health check (Section 3.8)."""

    backend: str
    reachable: bool
    detail: str = ""


class GenerationBackendError(Exception):
    """Base exception for adapter failures.

    Subclasses let resilience.py (built in G2) distinguish RPM/TPM errors
    (worth retrying) from RPD/quota errors (worth failing over instead) -
    see Section 3.6. We define the taxonomy now so G1/G2 adapters raise
    the right one from day one, even though nothing catches these
    specifically until G2.
    """


class RateLimitedError(GenerationBackendError):
    """HTTP 429 / rolling-window rate limit (RPM or TPM). Retry with backoff."""


class QuotaExceededError(GenerationBackendError):
    """Hard daily cap (RPD) or account quota. Do NOT retry - fail over instead."""


class BackendAuthError(GenerationBackendError):
    """The backend rejected our credentials - not a rate limit, don't retry."""


class BackendUnavailableError(GenerationBackendError):
    """The backend itself is unusable for reasons unrelated to rate limits,
    quota, or credentials - e.g. a local Ollama process that isn't running,
    a model that was never pulled, or the host being unreachable.

    Flagged as a real gap during G1 (see the project progress log, Session
    7): RateLimitedError/QuotaExceededError/BackendAuthError all assume a
    remote, metered, authenticated API, which doesn't describe a local
    process's failure modes. This is a setup/environment problem, not a
    transient one - resilience.py (G2) treats it like BackendAuthError:
    never retry the SAME backend for it, but still allow failover to the
    NEXT configured backend, since the problem is specific to this one.
    """


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
        """Cheap reachability check - see Section 3.8 for per-backend semantics.

        Must NOT be a full generation call. For hosted APIs this should be
        a minimal ping/models-list request; for local backends, a check
        that the process is up and the model is loaded.
        """
        raise NotImplementedError