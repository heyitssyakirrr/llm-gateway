"""
HTTP-facing response models for the observability routes (Phase G2).

Kept separate from any one capability's schemas/ folder (Section 4's
per-capability convention) since these report across ALL capabilities at
once - same reasoning as this module living outside capabilities/.
"""

from pydantic import BaseModel


class BackendModelInfo(BaseModel):
    """One row of GET /v1/models: what's configured, and whether it's
    actually usable right now."""

    capability: str
    backend: str
    model_name: str
    reachable: bool
    detail: str = ""


class ModelsResponse(BaseModel):
    models: list[BackendModelInfo]


class BackendStats(BaseModel):
    """One row of GET /v1/stats: aggregated request_log data for one
    (capability, backend) pair."""

    capability: str
    backend: str
    total_requests: int
    success_count: int
    error_count: int
    success_rate: float
    avg_latency_ms: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_estimate: float
    total_retries: int


class StatsResponse(BaseModel):
    stats: list[BackendStats]