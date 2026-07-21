"""
Resilience layer (Phase G2): exponential backoff + jitter for transient
rate limits, and failover across a configured backend order for anything
retrying won't fix.

This is the ONE place backoff/retry/failover logic lives, per the project
plan's Section 9 promise that adding a backend is "one adapter class, one
registry line" - if this logic were duplicated per-adapter or per-route,
every future capability (embed, rerank) would need to reinvent it, and
every backend would need to reimplement it. Instead, `router.py` builds an
ordered list of backend names to try, and `run_with_resilience` is the
only thing that ever calls `GenerationBackend.generate()` on the hot path.

How each declared error (base.py) is handled here is a deliberate policy
decision, not an accident of what happened to be easy to catch:

- RateLimitedError (RPM/TPM):
    Rolling-window limits clear on their own within seconds, so retrying
    the SAME backend with exponential backoff + full jitter is the
    correct response. After `max_retries_per_backend` attempts still
    fail, this backend is treated as effectively unavailable right now -
    fail over to the next one rather than keep waiting.

- QuotaExceededError (RPD / daily cap):
    A daily quota does not clear until the provider's reset, so retrying
    is pure wasted latency. Move to the next backend immediately.

- BackendAuthError / BackendUnavailableError:
    Config or environment problems (bad key, model not pulled, service
    down). Retrying the same backend would just repeat the same failure,
    so these never get a backoff attempt - but the problem is specific to
    THIS backend, so failover to the next one is still attempted.

- Any other declared GenerationBackendError (e.g. "this backend doesn't
  support image input"):
    Not retryable on this backend, but also not fatal for the whole
    request - try the next backend.

- Anything NOT a GenerationBackendError:
    An adapter is only supposed to raise GenerationBackendError (or a
    subclass) for calls that reached the provider; anything else (a bug,
    an unhandled exception type) is let through immediately rather than
    silently absorbed into a "try the next backend" path that would mask
    it as ordinary unreliability.
"""

import asyncio
import random

from app.capabilities.generate.base import (
    BackendAuthError,
    BackendUnavailableError,
    GenerationBackend,
    GenerationBackendError,
    GenerationParams,
    GenerationResult,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.generate.registry import GenerationRegistry


class ResilientCallResult:
    """What `run_with_resilience` hands back on success."""

    __slots__ = ("result", "backend_used", "fallback_chain", "retries")

    def __init__(
        self,
        result: GenerationResult,
        backend_used: GenerationBackend,
        fallback_chain: list[str],
        retries: int,
    ) -> None:
        self.result = result
        self.backend_used = backend_used
        # Every OTHER backend that was tried and failed before this one
        # succeeded - empty if the first attempt in `backend_order` worked.
        self.fallback_chain = fallback_chain
        # Total backoff-retry attempts across ALL backends tried, not just
        # the one that ultimately succeeded - this is what request_log's
        # `retries` column and the response envelope's `retries` field
        # actually mean (Section 3.5/5).
        self.retries = retries


class AllBackendsFailedError(Exception):
    """Raised when every backend in the attempt order failed and none
    could serve the request.

    Carries enough detail (`attempted`, `last_error`, `retries`) for the
    caller (routes.py) to log a rich failure entry and pick an honest
    status code, instead of re-deriving that from a generic exception.
    """

    def __init__(self, attempted: list[str], last_error: Exception, retries: int) -> None:
        self.attempted = attempted
        self.last_error = last_error
        self.retries = retries
        super().__init__(
            f"All configured backends failed after trying {attempted}: {last_error}"
        )


def _backoff_delay(attempt: int, base_seconds: float, max_seconds: float) -> float:
    """Exponential backoff with FULL jitter (the formula AWS's architecture
    blog recommends for this exact problem): a random delay in
    [0, min(max_seconds, base_seconds * 2**attempt)), not a fixed
    exponential value.

    Full jitter matters here specifically because a rate limit is a
    SHARED resource - if every retrying caller waited the same fixed
    delay, they'd all retry in lockstep and re-trigger the same limit
    (a thundering herd). A random delay spreads retries out instead.
    """
    ceiling = min(max_seconds, base_seconds * (2**attempt))
    return random.uniform(0, ceiling)


async def run_with_resilience(
    registry: GenerationRegistry,
    backend_order: list[str],
    params: GenerationParams,
    *,
    max_retries_per_backend: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> ResilientCallResult:
    """Attempt `params` against each backend in `backend_order`, in order,
    applying backoff-retry for RateLimitedError and moving on to the next
    backend for anything else declared in the error taxonomy.

    Raises AllBackendsFailedError if every backend in `backend_order` was
    tried and none succeeded.
    """
    fallback_chain: list[str] = []
    total_retries = 0
    last_exc: Exception | None = None

    for position, backend_name in enumerate(backend_order):
        backend = registry.get(backend_name)
        if position > 0:
            fallback_chain.append(backend_name)

        attempt = 0
        while True:
            try:
                result = await backend.generate(params)
                return ResilientCallResult(
                    result=result,
                    backend_used=backend,
                    fallback_chain=fallback_chain,
                    retries=total_retries,
                )
            except RateLimitedError as exc:
                last_exc = exc
                if attempt >= max_retries_per_backend:
                    # Still rate-limited after N backoff attempts - in
                    # practice this backend is unavailable right now.
                    # Stop retrying it and fail over.
                    break
                delay = _backoff_delay(attempt, base_delay_seconds, max_delay_seconds)
                await asyncio.sleep(delay)
                attempt += 1
                total_retries += 1
                continue
            except QuotaExceededError as exc:
                # Daily cap - will not clear before the provider's reset.
                # Retrying is pointless; move on immediately.
                last_exc = exc
                break
            except (BackendAuthError, BackendUnavailableError) as exc:
                # Config/environment problem specific to this backend -
                # never retry it for this, but still try the next one.
                last_exc = exc
                break
            except GenerationBackendError as exc:
                # Any other declared adapter failure (e.g. an unsupported
                # input for this backend) - not retryable here, try next.
                last_exc = exc
                break

    raise AllBackendsFailedError(
        attempted=list(backend_order),
        last_error=last_exc or RuntimeError("No generation backend was configured to try."),
        retries=total_retries,
    )