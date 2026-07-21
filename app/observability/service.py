"""
Observability service (Phase G2): builds the data behind GET /v1/models
and GET /v1/stats.

Deliberately kept OUT of capabilities/ - Section 4's per-capability
folders each own one AI capability (generate, embed, rerank); this module
is cross-cutting (it reports on ALL of them at once), same reasoning as
/v1/health living in main.py rather than any one capability's routes.py.
"""

import asyncio

from app.capabilities.generate.registry import GenerationRegistry
from app.logging_db import fetch_backend_stats
from app.observability.schemas import (
    BackendModelInfo,
    BackendStats,
    ModelsResponse,
    StatsResponse,
)


async def build_models_response(generation_registry: GenerationRegistry) -> ModelsResponse:
    """Reports every configured generation backend, its currently
    configured model name, and whether it's reachable right now.

    Free-tier model catalogs shift without notice (the project plan's own
    G0 notes cite a Gemini 404 from exactly this) - this is what lets a
    caller check "is this backend even usable today" without reading
    config files or adapter source.

    Extending to embed/rerank (G3/G4): add that capability's registry as
    a parameter, gather its health the same way, and extend `models` -
    the route and response schema don't need to change shape.
    """
    backends = generation_registry.all()
    health_results = await asyncio.gather(*(b.health_check() for b in backends))

    models = [
        BackendModelInfo(
            capability="generate",
            backend=backend.name,
            model_name=getattr(backend, "model_name", "unknown"),
            reachable=health.reachable,
            detail=health.detail or "",
        )
        for backend, health in zip(backends, health_results, strict=True)
    ]
    return ModelsResponse(models=models)


def build_stats_response() -> StatsResponse:
    """Aggregates request_log into per-(capability, backend) stats.

    Read-only, backed entirely by the request_log rows every route has
    been writing since G0 - no new logging call sites needed anywhere.
    """
    rows = fetch_backend_stats()
    stats = [
        BackendStats(
            capability=row["capability"],
            backend=row["backend_used"],
            total_requests=row["total_requests"],
            success_count=row["success_count"] or 0,
            error_count=row["total_requests"] - (row["success_count"] or 0),
            success_rate=(
                (row["success_count"] or 0) / row["total_requests"]
                if row["total_requests"]
                else 0.0
            ),
            avg_latency_ms=row["avg_latency_ms"],
            total_prompt_tokens=row["total_prompt_tokens"] or 0,
            total_completion_tokens=row["total_completion_tokens"] or 0,
            total_cost_estimate=row["total_cost_estimate"] or 0.0,
            total_retries=row["total_retries"] or 0,
        )
        for row in rows
    ]
    return StatsResponse(stats=stats)