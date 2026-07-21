"""
Route handler for the extract capability - schema-enforced structured output.

/v1/extract reuses the SAME generation backends as /v1/generate (the
identical GenerationRegistry, the identical adapters, and, as of G2, the
identical resilience layer). Extraction isn't a different kind of model
call - it's a generation call with a JSON-Schema contract wrapped around
the output: build a prompt telling the model to return matching JSON,
then parse + validate the response and retry (centrally, here, once) if
it doesn't comply.

Two DIFFERENT kinds of retry happen in this route, and they're kept
deliberately separate:
  - Backend-level retry/failover (rate limits, quota, outages) is handled
    entirely by `route_generation` -> resilience.py, exactly as it is for
    /v1/generate. This route never sees a raw backend call fail and retry
    it itself.
  - Schema-level retry (the model responded, but the JSON was malformed
    or didn't match the schema) is this route's own concern - it's an
    extraction-specific failure mode that has nothing to do with which
    backend served the request, so it doesn't belong in resilience.py.
"""

import asyncio
import time
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.auth import verify_api_key
from app.capabilities.extract.prompting import build_extraction_instruction
from app.capabilities.extract.validator import ExtractionParseError, parse_and_validate
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
from app.schemas.extract import ExtractRequest, ExtractResponse

router = APIRouter()


def _get_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


def _classify_error_type(exc: Exception) -> str:
    """Same taxonomy as /v1/generate's, plus schema_invalid - the one
    failure mode that's unique to this capability."""
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
    if isinstance(exc, ExtractionParseError):
        return "schema_invalid"
    return "other"


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
    x_request_id: str | None = Header(default=None),
) -> ExtractResponse:
    request_id = x_request_id or str(uuid.uuid4())
    registry = _get_registry(http_request)
    settings = get_settings()

    # Metadata only (Section 7) - notably, the schema itself is NOT logged
    # here even though it's structural rather than user content, to stay
    # conservative: a caller's schema could describe sensitive fields
    # (e.g. "ssn", "diagnosis") even with no real values attached.
    params_used = {
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "max_retries": body.max_retries,
        "has_image": body.image_base64 is not None,
        "image_mime_type": body.image_mime_type,
    }

    schema_instruction = build_extraction_instruction(body.json_schema)
    system_instruction = (
        f"{body.system_instruction}\n\n{schema_instruction}"
        if body.system_instruction
        else schema_instruction
    )

    start = time.perf_counter()
    attempt = 0  # schema retries used so far - NOT backend-level retries
    last_parse_error: ExtractionParseError | None = None
    result = None
    parsed: dict | None = None
    current_prompt = body.prompt
    backend_used_name: str | None = None
    fallback_chain: list[str] = []

    while True:
        gen_params = GenerationParams(
            prompt=current_prompt,
            system_instruction=system_instruction,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            image_base64=body.image_base64,
            image_mime_type=body.image_mime_type,
        )

        try:
            decision = await route_generation(registry, settings, body.backend, gen_params)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RouterConfigError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except AllBackendsFailedError as exc:
            # A genuine backend failure (resilience already retried and
            # failed over as far as it could) is NOT a "malformed JSON,
            # try again" situation - it's reported immediately, exactly
            # like /v1/generate does, rather than consuming a schema-retry
            # attempt on a problem schema-retry can't fix.
            latency_ms = int((time.perf_counter() - start) * 1000)
            error_type = _classify_error_type(exc.last_error)
            log_request(
                RequestLogEntry(
                    request_id=request_id,
                    caller_id=caller_id,
                    capability="extract",
                    endpoint="/v1/extract",
                    backend_requested=body.backend,
                    backend_used=None,
                    model_name=None,
                    params_used=params_used,
                    fallback_chain=exc.attempted[1:],
                    latency_ms=latency_ms,
                    retries=attempt,
                    success=False,
                    error_type=error_type,
                )
            )
            status_code = 429 if error_type in ("rpm_tpm", "rpd_quota") else 502
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

        result = decision.result
        backend_used_name = decision.backend_used.name
        fallback_chain = decision.fallback_chain

        try:
            parsed = parse_and_validate(result.text, body.json_schema)
            break  # success - `attempt` already equals schema retries used to get here
        except ExtractionParseError as exc:
            last_parse_error = exc
            if attempt >= body.max_retries:
                break  # give up - `attempt` equals retries used (== max_retries)
            attempt += 1
            # Feed the specific error back to the model - this is *why*
            # centralizing retry here (rather than each caller writing its
            # own loop) is actually valuable: the retry prompt can name
            # exactly what was wrong, not just "try again".
            current_prompt = (
                f"{body.prompt}\n\n"
                f"Your previous response was invalid: {exc}\n"
                "Respond again with ONLY the corrected JSON."
            )

    latency_ms = int((time.perf_counter() - start) * 1000)

    if parsed is None:
        # Exhausted every schema retry without ever producing valid,
        # schema-matching JSON. The backend(s) responded fine - the
        # content just never satisfied the contract - so this is a 422,
        # not a 502/429.
        log_request(
            RequestLogEntry(
                request_id=request_id,
                caller_id=caller_id,
                capability="extract",
                endpoint="/v1/extract",
                backend_requested=body.backend,
                backend_used=backend_used_name,
                model_name=result.model_name if result else None,
                params_used=params_used,
                fallback_chain=fallback_chain,
                latency_ms=latency_ms,
                retries=attempt,
                success=False,
                error_type="schema_invalid",
            )
        )
        raise HTTPException(
            status_code=422,
            detail=f"Model output did not satisfy the schema after {attempt + 1} attempt(s) "
            f"({attempt} retries): {last_parse_error}",
        )

    cost = estimate_cost(
        backend_used_name, result.model_name, result.prompt_tokens, result.completion_tokens
    )
    tokens_used = None
    if result.prompt_tokens is not None and result.completion_tokens is not None:
        tokens_used = result.prompt_tokens + result.completion_tokens

    log_request(
        RequestLogEntry(
            request_id=request_id,
            caller_id=caller_id,
            capability="extract",
            endpoint="/v1/extract",
            backend_requested=body.backend,
            backend_used=backend_used_name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=fallback_chain,
            latency_ms=latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_estimate=cost,
            retries=attempt,
            success=True,
        )
    )

    return ExtractResponse(
        data=parsed,
        backend_used=backend_used_name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_estimate=cost,
        retries=attempt,
    )