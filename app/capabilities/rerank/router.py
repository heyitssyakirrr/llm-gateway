"""
Router for the rerank capability. Structurally identical to
generate/router.py and embed/router.py - same job (resolve attempt
order, delegate to the shared resilience engine), same shape.

Worth noting explicitly since it's a genuine edge case: with only one
configured backend (Section 3.3 - no free-tier fallback provider exists
for reranking), `build_attempt_order` returns a list with exactly one
entry, and `run_with_resilience`'s `for backend_name in backend_order`
loop simply runs once - no special-casing was needed anywhere in the
shared resilience code to support a single-backend capability. If a
second rerank provider is ever added, it slots into
`RERANK_FALLBACK_ORDER` exactly like a second generate/embed backend
would - nothing about this router or the resilience engine changes.
"""

from app.capabilities.common.resilience import (
    ResilientCallResult,
    RouterConfigError,
    build_attempt_order,
    run_with_resilience,
)
from app.capabilities.rerank.base import RerankParams
from app.capabilities.rerank.registry import RerankRegistry
from app.config import Settings

__all__ = ["RouterConfigError", "route_rerank"]


async def route_rerank(
    registry: RerankRegistry,
    settings: Settings,
    requested_backend: str | None,
    params: RerankParams,
) -> ResilientCallResult:
    """Resolve the attempt order for a rerank call and run it through the
    resilience layer. Same three-exception contract as route_generation
    and route_embedding - see route_generation's docstring."""
    known = registry.names()
    if requested_backend is not None and requested_backend not in known:
        raise KeyError(
            f"Unknown rerank backend '{requested_backend}'. Known backends: {known}"
        )

    effective_primary = requested_backend or settings.rerank_primary_backend
    if effective_primary not in known:
        raise RouterConfigError(
            f"Configured primary backend '{effective_primary}' is not a registered "
            f"rerank backend. Known backends: {known}. Check RERANK_PRIMARY_BACKEND."
        )

    attempt_order = build_attempt_order(effective_primary, settings.rerank_fallback_order, known)

    return await run_with_resilience(
        registry=registry,
        backend_order=attempt_order,
        call=lambda backend: backend.rerank(params),
        max_retries_per_backend=settings.rerank_max_retries_per_backend,
        base_delay_seconds=settings.rerank_backoff_base_seconds,
        max_delay_seconds=settings.rerank_backoff_max_seconds,
    )
