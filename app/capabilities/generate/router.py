"""
Router (Section 4, G2): decides the ORDER of backends to attempt for a
/v1/generate (and /v1/extract) call, then delegates the actual
call-with-backoff-and-failover mechanics to resilience.py.

G0/G1 shape: resolve one backend, no failover. G2 shape: build an ordered
attempt list - the requested/primary backend first, then every OTHER
backend from the configured fallback order - and hand it to
`run_with_resilience`. This keeps "which backend to try in what order"
(a routing/config concern) separate from "what to do when an attempt
fails" (a resilience concern), so either can change independently.
"""

from app.capabilities.generate.base import GenerationParams
from app.capabilities.generate.registry import GenerationRegistry
from app.capabilities.generate.resilience import ResilientCallResult, run_with_resilience
from app.config import Settings


def _build_attempt_order(
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
    silently dropped rather than allowed to reach resilience.py - this is
    what keeps a config typo or a not-yet-deployed backend name from
    crashing every request with a raw KeyError (see the docstring on
    `RouterConfigError` for why this is a distinct error class from a
    caller pinning an unknown backend).
    """
    rest = [name for name in configured_fallback_order if name != effective_primary and name in known]
    return [effective_primary, *rest]


class RouterConfigError(Exception):
    """The configured primary/fallback backend names don't match what's
    actually registered - an operator/deployment problem (a typo in
    GENERATION_FALLBACK_ORDER, or a backend removed from the registry
    without updating config), never a caller error.

    Deliberately a DIFFERENT exception type than KeyError (which routes.py
    maps to 400 "you asked for a backend that doesn't exist"): this one
    means the request itself was fine, but the *server's own configuration*
    is broken, which is a 500, not a 400 - the caller didn't do anything
    wrong here.
    """


async def route_generation(
    registry: GenerationRegistry,
    settings: Settings,
    requested_backend: str | None,
    params: GenerationParams,
) -> ResilientCallResult:
    """Resolve the attempt order for a generation call and run it through
    the resilience layer.

    Raises:
        KeyError: `requested_backend` was given but isn't a configured
            backend name - a client error (400), not a resilience concern.
        RouterConfigError: the configured primary backend isn't a
            registered backend - a server-side config error (500), never
            the caller's fault.
        AllBackendsFailedError: every backend in the attempt order failed.
    """
    known = registry.names()
    if requested_backend is not None and requested_backend not in known:
        raise KeyError(
            f"Unknown generation backend '{requested_backend}'. Known backends: {known}"
        )

    effective_primary = requested_backend or settings.generation_primary_backend
    if effective_primary not in known:
        raise RouterConfigError(
            f"Configured primary backend '{effective_primary}' is not a registered "
            f"generation backend. Known backends: {known}. Check GENERATION_PRIMARY_BACKEND."
        )

    attempt_order = _build_attempt_order(effective_primary, settings.generation_fallback_order, known)

    return await run_with_resilience(
        registry=registry,
        backend_order=attempt_order,
        params=params,
        max_retries_per_backend=settings.generation_max_retries_per_backend,
        base_delay_seconds=settings.generation_backoff_base_seconds,
        max_delay_seconds=settings.generation_backoff_max_seconds,
    )