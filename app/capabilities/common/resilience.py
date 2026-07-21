"""
Resilience layer, shared across every capability that makes real
provider calls (generate, embed, and rerank later).

This used to live inside `capabilities/generate/resilience.py`. G3 needed
the identical backoff/retry/failover mechanics for embedding calls, and
this module's own original docstring already said the quiet part out
loud: "every future capability (embed, rerank) would need to reinvent it"
if this logic stayed generate-specific. So it moved here, and the one
thing that had to change to make that true is `run_with_resilience` no
longer hardcodes `backend.generate(params)` - it takes a `call` function
instead, supplied by whichever capability's router is using it:

    # generate/router.py
    run_with_resilience(registry, order, call=lambda b: b.generate(params), ...)

    # embed/router.py
    run_with_resilience(registry, order, call=lambda b: b.embed(params), ...)

Everything else - the retry-vs-failover POLICY per error type - is
unchanged and, more importantly, now genuinely enforced identically for
every capability instead of by convention/copy-paste:

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

- Any other declared BackendError (e.g. "this backend doesn't support
  image input"):
    Not retryable on this backend, but also not fatal for the whole
    request - try the next backend.

- Anything NOT a BackendError:
    An adapter is only supposed to raise BackendError (or a subclass) for
    calls that reached the provider; anything else (a bug, an unhandled
    exception type) is let through immediately rather than silently
    absorbed into a "try the next backend" path that would mask it as
    ordinary unreliability.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from app.capabilities.common.errors import (
    BackendAuthError,
    BackendError,
    BackendUnavailableError,
    QuotaExceededError,
    RateLimitedError,
)

BackendT = TypeVar("BackendT")
ResultT = TypeVar("ResultT")


class BackendRegistry(Protocol[BackendT]):
    """The only shape resilience.py needs from a registry. Both
    `GenerationRegistry` and `EmbeddingRegistry` satisfy this - Python's
    `Protocol` matches structurally (by having a `.get()`/`.names()`),
    not by declared inheritance - so neither registry has to import from
    or subclass anything in `common/` to be usable here."""

    def get(self, name: str) -> BackendT: ...
    def names(self) -> list[str]: ...


class RouterConfigError(Exception):
    """The configured primary/fallback backend names don't match what's
    actually registered - an operator/deployment problem (a typo in a
    fallback-order env var, or a backend removed from a registry without
    updating config), never a caller error.

    Deliberately a DIFFERENT exception type than KeyError (which each
    capability's routes.py maps to 400 "you asked for a backend that
    doesn't exist"): this one means the request itself was fine, but the
    *server's own configuration* is broken, which is a 500, not a 400.
    """


def build_attempt_order(
    effective_primary: str,
    configured_fallback_order: list[str],
    known: list[str],
) -> list[str]:
    """`effective_primary` is always tried first, regardless of where (or
    whether) it appears in `configured_fallback_order` - a caller who
    pinned a backend, or the operator-configured default primary, should
    always get first crack at serving the request. Every other backend
    from the configured order follows, in that order, as failover targets.

    Entries in `configured_fallback_order` that aren't in `known` are
    silently dropped rather than allowed to crash every request with a
    raw KeyError - see `RouterConfigError`'s docstring for why a bad
    *configured* name is handled separately from a caller pinning an
    unknown one.
    """
    rest = [name for name in configured_fallback_order if name != effective_primary and name in known]
    return [effective_primary, *rest]


class ResilientCallResult:
    """What `run_with_resilience` hands back on success."""

    __slots__ = ("result", "backend_used", "fallback_chain", "retries")

    def __init__(
        self,
        result: object,
        backend_used: object,
        fallback_chain: list[str],
        retries: int,
    ) -> None:
        self.result = result
        # Every OTHER backend that was tried and failed before this one
        # succeeded - empty if the first attempt in `backend_order` worked.
        self.fallback_chain = fallback_chain
        self.backend_used = backend_used
        # Total backoff-retry attempts across ALL backends tried, not just
        # the one that ultimately succeeded - this is what request_log's
        # `retries` column and the response envelope's `retries` field
        # actually mean.
        self.retries = retries


class AllBackendsFailedError(Exception):
    """Raised when every backend in the attempt order failed and none
    could serve the request.

    Carries enough detail (`attempted`, `last_error`, `retries`) for the
    caller (a capability's routes.py) to log a rich failure entry and
    pick an honest status code, instead of re-deriving that from a
    generic exception.
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
    registry: BackendRegistry[BackendT],
    backend_order: list[str],
    call: Callable[[BackendT], Awaitable[ResultT]],
    *,
    max_retries_per_backend: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> ResilientCallResult:
    """Attempt `call(backend)` against each backend in `backend_order`, in
    order, applying backoff-retry for RateLimitedError and moving on to
    the next backend for anything else declared in the error taxonomy.

    `call` is the one thing that makes this capability-agnostic: it's a
    small function supplied by the caller (a capability's router.py) that
    knows how to invoke *that* capability's method on a backend - e.g.
    `lambda b: b.generate(params)` or `lambda b: b.embed(params)`. This
    function never needs to know which one it's calling.

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
                result = await call(backend)
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
            except BackendError as exc:
                # Any other declared adapter failure (e.g. an unsupported
                # input for this backend) - not retryable here, try next.
                last_exc = exc
                break

    raise AllBackendsFailedError(
        attempted=list(backend_order),
        last_error=last_exc or RuntimeError("No backend was configured to try."),
        retries=total_retries,
    )
