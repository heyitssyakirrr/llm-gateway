"""
FastAPI app - route handlers only. No provider-specific logic lives here
(Section 4): everything provider-specific is in backends/, everything
about "which backend" is in router.py/registry.py, everything about
logging is in logging_db.py. This file just wires them together.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.auth import verify_api_key
from app.backends.base import (
    BackendAuthError,
    GenerationParams,
    QuotaExceededError,
    RateLimitedError,
)
from app.config import get_settings
from app.logging_db import RequestLogEntry, init_db, log_request
from app.pricing import estimate_cost
from app.registry import GenerationRegistry, build_generation_registry
from app.router import route_generation
from app.schemas import BackendHealth, GenerateRequest, GenerateResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.generation_registry = build_generation_registry(get_settings())
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


def _get_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


def _classify_error_type(exc: Exception) -> str:
    """Map a raised exception onto request_log's error_type taxonomy
    (Section 5): rpm_tpm / rpd_quota / auth / timeout / other."""
    if isinstance(exc, RateLimitedError):
        return "rpm_tpm"
    if isinstance(exc, QuotaExceededError):
        return "rpd_quota"
    if isinstance(exc, BackendAuthError):
        return "auth"
    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return "timeout"
    return "other"


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    body: GenerateRequest,
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
    x_request_id: str | None = Header(default=None),
) -> GenerateResponse:
    request_id = x_request_id or str(uuid.uuid4())
    registry = _get_registry(http_request)

    # Metadata only - Section 7's no-content-logging rule. Never put
    # `prompt` or `system_instruction` in here.
    params_used = {
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "stop_sequences": body.stop_sequences,
    }

    try:
        decision = route_generation(registry, body.backend)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    gen_params = GenerationParams(
        prompt=body.prompt,
        system_instruction=body.system_instruction,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p,
        top_k=body.top_k,
        stop_sequences=body.stop_sequences,
    )

    start = time.perf_counter()
    try:
        result = await decision.backend.generate(gen_params)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        error_type = _classify_error_type(exc)
        log_request(
            RequestLogEntry(
                request_id=request_id,
                caller_id=caller_id,
                capability="generate",
                endpoint="/v1/generate",
                backend_requested=body.backend,
                backend_used=decision.backend.name,
                model_name=None,
                params_used=params_used,
                fallback_chain=decision.fallback_chain,
                latency_ms=latency_ms,
                success=False,
                error_type=error_type,
            )
        )
        status_code = 429 if error_type in ("rpm_tpm", "rpd_quota") else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    cost = estimate_cost(
        decision.backend.name, result.model_name, result.prompt_tokens, result.completion_tokens
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
            backend_used=decision.backend.name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=decision.fallback_chain,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_estimate=cost,
            success=True,
        )
    )

    return GenerateResponse(
        data=result.text,
        backend_used=decision.backend.name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_estimate=cost,
        retries=0,
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health(
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
) -> HealthResponse:
    registry = _get_registry(http_request)
    results = await asyncio.gather(*(b.health_check() for b in registry.all()))
    return HealthResponse(
        backends=[BackendHealth(backend=r.backend, reachable=r.reachable, detail=r.detail) for r in results]
    )
