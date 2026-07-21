"""
Shared backend error taxonomy.

G0/G1 originally defined this taxonomy inside `capabilities/generate/base.py`,
which was fine while generation was the only capability that made real
provider calls. G3 (embedding) needs the exact same distinctions - a
Cohere 429 is still "retry with backoff" vs "fail over now" in the same
way a Gemini 429 is - so this lives here now, capability-agnostic, and
both `generate/base.py` and `embed/base.py` import from here instead of
duplicating it.

`GenerationBackendError` is kept as an alias of `BackendError` (same
class object, not a copy) purely so every pre-existing
`from app.capabilities.generate.base import GenerationBackendError`
import elsewhere in the codebase keeps working unchanged - `isinstance`
checks against either name still succeed.
"""


class BackendError(Exception):
    """Base exception for adapter failures, across every capability.

    Subclasses let `common/resilience.py` distinguish RPM/TPM errors
    (worth retrying) from RPD/quota errors (worth failing over instead).
    """


class RateLimitedError(BackendError):
    """HTTP 429 / rolling-window rate limit (RPM or TPM). Retry with backoff."""


class QuotaExceededError(BackendError):
    """Hard daily cap (RPD) or account quota. Do NOT retry - fail over instead."""


class BackendAuthError(BackendError):
    """The backend rejected our credentials - not a rate limit, don't retry."""


class BackendUnavailableError(BackendError):
    """The backend itself is unusable for reasons unrelated to rate limits,
    quota, or credentials - e.g. a local Ollama process that isn't running,
    a model that was never pulled, or the host being unreachable.

    Not retryable on the same backend, but the problem is specific to
    THIS backend, so failover to the next one is still attempted.
    """


# Backward-compatible alias - see module docstring. Existing imports of
# `GenerationBackendError` from `capabilities/generate/base.py` continue
# to resolve to this exact class.
GenerationBackendError = BackendError
