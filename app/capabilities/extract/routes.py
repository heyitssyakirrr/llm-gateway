"""
Route handler for the extract capability - schema-enforced structured output.

/v1/extract reuses the SAME generation backends as /v1/generate (the
identical GenerationRegistry, the identical adapters). Extraction isn't a
different kind of model call - it's a generation call with a JSON-Schema
contract wrapped around the output: build a prompt telling the model to
return matching JSON, then parse + validate the response and retry
(centrally, here, once) if it doesn't comply. No adapter needs to know
extraction exists - that's the whole point of keeping this cross-cutting
concern out of backends/.
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
    GenerationParams,
    GenerationResult,
    QuotaExceededError,
    RateLimitedError,
)
from app.capabilities.generate.registry import GenerationRegistry
from app.capabilities.generate.router import route_generation
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

    try:
        decision = route_generation(registry, body.backend)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    schema_instruction = build_extraction_instruction(body.json_schema)
    system_instruction = (
        f"{body.system_instruction}\n\n{schema_instruction}"
        if body.system_instruction
        else schema_instruction
    )

    start = time.perf_counter()
    attempt = 0  # retries USED so far (0 on the first/initial call, not a retry yet)
    last_parse_error: ExtractionParseError | None = None
    result: GenerationResult | None = None
    parsed: dict | None = None
    current_prompt = body.prompt

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
            result = await decision.backend.generate(gen_params)
        except Exception as exc:
            # A genuine backend failure (rate limit, auth, unreachable) is
            # NOT a "malformed JSON, try again" situation - retrying here
            # would blur schema-retry logic with backend-failure logic,
            # which belongs to G2's resilience layer instead. Fail
            # immediately, exactly like /v1/generate does.
            latency_ms = int((time.perf_counter() - start) * 1000)
            error_type = _classify_error_type(exc)
            log_request(
                RequestLogEntry(
                    request_id=request_id,
                    caller_id=caller_id,
                    capability="extract",
                    endpoint="/v1/extract",
                    backend_requested=body.backend,
                    backend_used=decision.backend.name,
                    model_name=None,
                    params_used=params_used,
                    fallback_chain=decision.fallback_chain,
                    latency_ms=latency_ms,
                    retries=attempt,
                    success=False,
                    error_type=error_type,
                )
            )
            status_code = 429 if error_type in ("rpm_tpm", "rpd_quota") else 502
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

        try:
            parsed = parse_and_validate(result.text, body.json_schema)
            break  # success - `attempt` already equals retries used to get here
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
        # Exhausted every retry without ever producing valid, schema-
        # matching JSON. This is the caller's signal that this backend/
        # model genuinely can't do this extraction reliably - a 422, not
        # a 502, since the backend responded fine, its content just never
        # satisfied the contract.
        log_request(
            RequestLogEntry(
                request_id=request_id,
                caller_id=caller_id,
                capability="extract",
                endpoint="/v1/extract",
                backend_requested=body.backend,
                backend_used=decision.backend.name,
                model_name=result.model_name if result else None,
                params_used=params_used,
                fallback_chain=decision.fallback_chain,
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
        decision.backend.name, result.model_name, result.prompt_tokens, result.completion_tokens
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
            backend_used=decision.backend.name,
            model_name=result.model_name,
            params_used=params_used,
            fallback_chain=decision.fallback_chain,
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
        backend_used=decision.backend.name,
        model_name=result.model_name,
        request_id=request_id,
        latency_ms=latency_ms,
        tokens_used=tokens_used,
        cost_estimate=cost,
        retries=attempt,
    )