from pydantic import BaseModel


class BackendHealth(BaseModel):
    backend: str
    reachable: bool
    detail: str = ""


class HealthResponse(BaseModel):
    """GET /v1/health - per-backend status, never one aggregate boolean
    (Section 3.8)."""

    backends: list[BackendHealth]
