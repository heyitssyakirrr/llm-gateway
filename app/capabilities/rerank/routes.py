"""
Route handler for the rerank capability (Section 4).

Structurally identical to embed/routes.py: backend selection, retry, and
failover all live behind `route_rerank` (router.py -> the shared
resilience engine in common/resilience.py) - this file's job is purely
HTTP concerns: validate the request, call the router, translate the
outcome into the standardized response envelope (Section 3.5), and log
exactly one row per request (Section 5).
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.auth import verify_api_key
from app.capabilities.common.resilience import AllBackendsFailedError
from app.capabilities.rerank.base import (
    BackendAuthError,
    BackendUnavailableError,
    QuotaExceededError,
    RateLimitedError,
    RerankParams,
)
from app.capabilities.rerank.registry import RerankRegistry
from app.capabilities.rerank.router import RouterConfigError, route_rerank
from app.config import get_settings
from app.logging_db import RequestLogEntry, log_request
from app.pricing import estimate_cost
from app.schemas.rerank import RerankedItemResponse, RerankRequest, RerankResponse

router = APIRouter()


def _get_registry(request: Request) -> RerankRegistry:
    return request.app.state.rerank_registry


def _classify_error_type(exc: Exception) -> str:
    """Same taxonomy as /v1/generate's and /v1/embed's _classify_error_type
    - see generate/routes.py for the full rationale. Duplicated rather
    than imported because each capability's routes.py owns its own
    HTTP-facing mapping, the same way each capability owns its own
    routes.py in general."""
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


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    body: RerankRequest,
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
    x_request_id: str | None = Header(default=None),
) -> RerankResponse:
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty or whitespace-only.")
    if body.has_blank_document:
        # Rejected here, not in the schema layer (see RerankRequest's
        # `has_blank_document` docstring) - an empty string is
        # structurally a valid list entry, but reranking against "" is
        # never a meaningful call, same reasoning as /v1/embed's
        # equivalent check.
        raise HTTPException(
            status_code=400, detail="documents must not contain empty or whitespace-only entries."
        )
    if body.top_n is not None and body.top_n > len(body.documents):
        # Not a provider-level error on any backend tested (Cohere just
        # clamps silently) - but silently returning fewer results than a
        # caller explicitly asked for is exactly the kind of "seems fine
        # until it quietly isn't" bug Section 7.5 exists to catch early,
        # so this is rejected here instead of passed through.
        raise HTTPException(
            status_code=400,
            detail=f"top_n ({body.top_n}) cannot exceed the number of documents ({len(body.documents)}).",
        )

    request_id = x_request_id or str(uuid.uuid4())
    registry = _get_registry(http_request)
    settings = get_settings()

    # Metadata only (Section 7) - the actual query/document content is
    # never logged, same rule as prompts on /v1/generate and text on
    # /v1/embed.
    params_used = {
        "document_count": len(body.documents),
        "top_n": body.top_n,
    }

    rerank_params = RerankParams(query=body.query, documents=body.documents, top_n=body.top_n)

    start = time.perf_counter()
    try:
        decision = await route_rerank(registry, settings, body.backend, rerank_params)
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
                capability="rerank",
                endpoint="/v1/rerank",
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
    cost = estimate_cost(backend_used.name, result.model_name, None, None)

    log_request(
        RequestLogEntry(
            request_id=request_id,
            caller_id=caller_id,
            capability="rerank",
            endpoint="/v1/rerank",
            backend_requested=body.backend,
            backend_used=backend_used.name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=decision.fallback_chain,
            latency_ms=latency_ms,
            prompt_tokens=None,
            completion_tokens=None,
            cost_estimate=cost,
            retries=decision.retries,
            success=True,
        )
    )

    return RerankResponse(
        data=[
            RerankedItemResponse(index=item.index, text=item.text, relevance_score=item.relevance_score)
            for item in result.results
        ],
        backend_used=backend_used.name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        cost_estimate=cost,
        retries=decision.retries,
    )
