"""
Router (Section 4): decides the ORDER of backends to attempt for a
/v1/generate (and /v1/extract) call, then delegates the actual
call-with-backoff-and-failover mechanics to the shared resilience engine
in `capabilities/common/resilience.py`.

G3 refactor: this file used to import `_build_attempt_order`,
`RouterConfigError`, and `run_with_resilience` from a generate-only
`resilience.py`. Those are now shared (`embed/router.py` uses the exact
same three things) - see `common/resilience.py`'s module docstring for
why. Nothing about this file's OWN job changed: it still just resolves
"which backend goes first, and what's the fallback order" (a routing
concern) and hands that off to the resilience engine (a separate
concern), so either can change independently.
"""

from app.capabilities.common.resilience import (
    ResilientCallResult,
    RouterConfigError,
    build_attempt_order,
    run_with_resilience,
)
from app.capabilities.generate.base import GenerationParams
from app.capabilities.generate.registry import GenerationRegistry
from app.config import Settings

# Re-exported so existing "from app.capabilities.generate.router import
# RouterConfigError" imports (generate/routes.py, extract/routes.py)
# keep working unchanged.
__all__ = ["RouterConfigError", "route_generation"]


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

    attempt_order = build_attempt_order(effective_primary, settings.generation_fallback_order, known)

    return await run_with_resilience(
        registry=registry,
        backend_order=attempt_order,
        call=lambda backend: backend.generate(params),
        max_retries_per_backend=settings.generation_max_retries_per_backend,
        base_delay_seconds=settings.generation_backoff_base_seconds,
        max_delay_seconds=settings.generation_backoff_max_seconds,
    )
