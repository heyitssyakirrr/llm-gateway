"""
Route handler for the generate capability (Section 4, G2).

Backend selection, retry, and failover all now live behind
`route_generation` (router.py -> resilience.py) - this file's job is
purely HTTP concerns: validate the request (Pydantic already did that),
call the router, translate the outcome into the standardized response
envelope (Section 3.5), and log exactly one row per request (Section 5)
regardless of whether it succeeded, retried, or failed over.
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.auth import verify_api_key
from app.capabilities.generate.base import (
    BackendAuthError,
    BackendUnavailableError,
    GenerationParams,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.common.resilience import AllBackendsFailedError
from app.capabilities.generate.registry import GenerationRegistry
from app.capabilities.generate.router import RouterConfigError, route_generation
from app.config import get_settings
from app.logging_db import RequestLogEntry, log_request
from app.pricing import estimate_cost
from app.schemas.generate import GenerateRequest, GenerateResponse

router = APIRouter()


def _get_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


def _classify_error_type(exc: Exception) -> str:
    """Map a raised exception onto request_log's error_type taxonomy
    (Section 5): rpm_tpm / rpd_quota / auth / unavailable / timeout / other.
    """
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


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
    x_request_id: str | None = Header(default=None),
) -> GenerateResponse:
    request_id = x_request_id or str(uuid.uuid4())
    registry = _get_registry(http_request)
    settings = get_settings()

    # Metadata only (Section 7) - never the prompt or response text itself.
    params_used = {
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "stop_sequences": body.stop_sequences,
        "has_image": body.image_base64 is not None,
        "image_mime_type": body.image_mime_type,
    }

    gen_params = GenerationParams(
        prompt=body.prompt,
        system_instruction=body.system_instruction,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        top_k=body.top_k,
        stop_sequences=body.stop_sequences,
        image_base64=body.image_base64,
        image_mime_type=body.image_mime_type,
    )

    start = time.perf_counter()
    try:
        decision = await route_generation(registry, settings, body.backend, gen_params)
    except KeyError as e:
        # Caller pinned a backend name we don't recognize - a client
        # error, resolved before any backend call was ever attempted.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RouterConfigError as e:
        # The SERVER's configuration is broken (e.g. GENERATION_PRIMARY_BACKEND
        # points at a backend that isn't registered) - never the caller's
        # fault, so this is a 500, not a 400.
        raise HTTPException(status_code=500, detail=str(e)) from e
    except AllBackendsFailedError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        error_type = _classify_error_type(exc.last_error)
        log_request(
            RequestLogEntry(
                request_id=request_id,
                caller_id=caller_id,
                capability="generate",
                endpoint="/v1/generate",
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
        # 429 when the underlying cause is rate/quota pressure across every
        # configured backend (the caller should back off and retry later);
        # 502 for everything else (auth/unavailable/other - an operator
        # problem, not something the caller can fix by retrying).
        status_code = 429 if error_type in ("rpm_tpm", "rpd_quota") else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    result = decision.result
    backend_used = decision.backend_used
    cost = estimate_cost(
        backend_used.name, result.model_name, result.prompt_tokens, result.completion_tokens
    )
    tokens_used = None
    if result.prompt_tokens is not None and result.completion_tokens is not None:
        tokens_used = result.prompt_tokens + result.completion_tokens

    log_request(
        RequestLogEntry(
            request_id=request_id,
            caller_id=caller_id,
            capability="generate",
            endpoint="/v1/generate",
            backend_requested=body.backend,
            backend_used=backend_used.name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=decision.fallback_chain,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_estimate=cost,
            retries=decision.retries,
            success=True,
        )
    )

    return GenerateResponse(
        data=result.text,
        backend_used=backend_used.name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_estimate=cost,
        retries=decision.retries,
    )