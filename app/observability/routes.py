"""
Observability routes (Phase G2): GET /v1/models, GET /v1/stats.

Both span every capability and every backend at once, so - like
/v1/health in main.py - they don't belong inside any one capability's
routes.py. Auth still applies (Depends(verify_api_key)): these expose
operational detail (which backends are configured, aggregate cost and
latency) that shouldn't be reachable without a valid key, same as every
other route.
"""

from fastapi import APIRouter, Depends, Request

from app.auth import verify_api_key
from app.capabilities.generate.registry import GenerationRegistry
from app.observability.schemas import ModelsResponse, StatsResponse
from app.observability.service import build_models_response, build_stats_response

router = APIRouter()


def _get_generation_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


@router.get("/v1/models", response_model=ModelsResponse)
async def get_models(
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
) -> ModelsResponse:
    registry = _get_generation_registry(http_request)
    return await build_models_response(registry)


@router.get("/v1/stats", response_model=StatsResponse)
async def get_stats(
    caller_id: str = Depends(verify_api_key),
) -> StatsResponse:
    return build_stats_response()