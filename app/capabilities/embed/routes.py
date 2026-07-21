"""
Route handler for the embed capability (Section 4).

Structurally identical to generate/routes.py: backend selection, retry,
and failover all live behind `route_embedding` (router.py -> the shared
resilience engine in common/resilience.py) - this file's job is purely
HTTP concerns: validate the request (Pydantic already did that), call the
router, translate the outcome into the standardized response envelope
(Section 3.5), and log exactly one row per request (Section 5).
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.auth import verify_api_key
from app.capabilities.common.resilience import AllBackendsFailedError
from app.capabilities.embed.base import (
    BackendAuthError,
    BackendUnavailableError,
    EmbeddingParams,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.embed.registry import EmbeddingRegistry
from app.capabilities.embed.router import RouterConfigError, route_embedding
from app.config import get_settings
from app.logging_db import RequestLogEntry, log_request
from app.pricing import estimate_cost
from app.schemas.embed import EmbedRequest, EmbedResponse

router = APIRouter()


def _get_registry(request: Request) -> EmbeddingRegistry:
    return request.app.state.embedding_registry


def _classify_error_type(exc: Exception) -> str:
    """Same taxonomy as /v1/generate's _classify_error_type - see that
    function for the full rationale. Duplicated rather than imported
    because each capability's routes.py owns its own HTTP-facing mapping,
    the same way each capability owns its own routes.py in general."""
    if isinstance(exc, RateLimitedError):
        return "rpm_tpm"
    if isinstance(exc, QuotaExceededError):
        return "rpd_quota"
    if isinstance(exc, BackendAuthError):
        return "auth"
    if isinstance(exc, BackendUnavailableError):
        return "unavailable"
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout"
    return "other"


@router.post("/v1/embed", response_model=EmbedResponse)
async def embed(
    body: EmbedRequest,
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
    x_request_id: str | None = Header(default=None),
) -> EmbedResponse:
    if body.has_blank_text:
        # Rejected here, not in the schema layer (see EmbedRequest's
        # `has_blank_text` docstring) - an empty string is structurally a
        # valid list entry, but embedding "" is never a meaningful call
        # on ANY backend, so this is a 400 before any backend is touched,
        # same "reject malformed requests early" principle as Section 7.5.
        raise HTTPException(
            status_code=400, detail="texts must not contain empty or whitespace-only entries."
        )

    request_id = x_request_id or str(uuid.uuid4())
    registry = _get_registry(http_request)
    settings = get_settings()

    # Metadata only (Section 7) - the actual text content is never
    # logged, same rule as prompts on /v1/generate. `text_count` lets
    # /v1/stats-style analysis later answer "how big are typical embed
    # batches" without ever touching content.
    params_used = {
        "task_type": body.task_type,
        "text_count": len(body.texts),
    }

    embed_params = EmbeddingParams(texts=body.texts, task_type=body.task_type)

    start = time.perf_counter()
    try:
        decision = await route_embedding(registry, settings, body.backend, embed_params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RouterConfigError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except AllBackendsFailedError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        error_type = _classify_error_type(exc.last_error)
        log_request(
            RequestLogEntry(
                request_id=request_id,
                caller_id=caller_id,
                capability="embed",
                endpoint="/v1/embed",
                backend_requested=body.backend,
                backend_used=None,
                model_name=None,
                params_used=params_used,
                fallback_chain=exc.attempted[1:],
                latency_ms=latency_ms,
                retries=exc.retries,
                success=False,
                error_type=error_type,
            )
        )
        status_code = 429 if error_type in ("rpm_tpm", "rpd_quota") else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    result = decision.result
    backend_used = decision.backend_used
    cost = estimate_cost(backend_used.name, result.model_name, result.total_tokens, None)

    log_request(
        RequestLogEntry(
            request_id=request_id,
            caller_id=caller_id,
            capability="embed",
            endpoint="/v1/embed",
            backend_requested=body.backend,
            backend_used=backend_used.name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=decision.fallback_chain,
            latency_ms=latency_ms,
            prompt_tokens=result.total_tokens,
            completion_tokens=None,
            cost_estimate=cost,
            retries=decision.retries,
            success=True,
        )
    )

    return EmbedResponse(
        data=result.vectors,
        dimensions=result.dimensions,
        backend_used=backend_used.name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        tokens_used=result.total_tokens,
        cost_estimate=cost,
        retries=decision.retries,
    )
