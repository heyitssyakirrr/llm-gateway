"""
Router for the embed capability. Structurally identical to
`capabilities/generate/router.py` - same job (resolve attempt order,
delegate to the shared resilience engine), same shape, different
capability. This is the real test of G2's design promise: if reusing
`run_with_resilience` here required changing `common/resilience.py`
itself, that would mean it was never actually generic - it isn't
changed at all, only `call=lambda backend: backend.embed(params)`
differs from generate's `lambda backend: backend.generate(params)`.
"""

from app.capabilities.common.resilience import (
    ResilientCallResult,
    RouterConfigError,
    build_attempt_order,
    run_with_resilience,
)
from app.capabilities.embed.base import EmbeddingParams
from app.capabilities.embed.registry import EmbeddingRegistry
from app.config import Settings

__all__ = ["RouterConfigError", "route_embedding"]


async def route_embedding(
    registry: EmbeddingRegistry,
    settings: Settings,
    requested_backend: str | None,
    params: EmbeddingParams,
) -> ResilientCallResult:
    """Resolve the attempt order for an embedding call and run it through
    the resilience layer. Same three-exception contract as
    `route_generation` - see that function's docstring."""
    known = registry.names()
    if requested_backend is not None and requested_backend not in known:
        raise KeyError(
            f"Unknown embedding backend '{requested_backend}'. Known backends: {known}"
        )

    effective_primary = requested_backend or settings.embedding_primary_backend
    if effective_primary not in known:
        raise RouterConfigError(
            f"Configured primary backend '{effective_primary}' is not a registered "
            f"embedding backend. Known backends: {known}. Check EMBEDDING_PRIMARY_BACKEND."
        )

    attempt_order = build_attempt_order(effective_primary, settings.embedding_fallback_order, known)

    return await run_with_resilience(
        registry=registry,
        backend_order=attempt_order,
        call=lambda backend: backend.embed(params),
        max_retries_per_backend=settings.embedding_max_retries_per_backend,
        base_delay_seconds=settings.embedding_backoff_base_seconds,
        max_delay_seconds=settings.embedding_backoff_max_seconds,
    )
