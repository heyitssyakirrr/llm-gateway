"""
FastAPI app entrypoint - creates the app, wires up lifespan/state, and
mounts each capability's router. No route logic lives here - that's each
capability's routes.py (Section 4). /v1/health stays here because it spans
every capability, not just one.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app.auth import verify_api_key
from app.capabilities.generate.registry import GenerationRegistry, build_generation_registry
from app.capabilities.generate.routes import router as generate_router
from app.capabilities.extract.routes import router as extract_router
from app.config import get_settings
from app.logging_db import init_db
from app.schemas.health import BackendHealth, HealthResponse


# things that happens once, when the app starts up, not once per request.  
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.generation_registry = build_generation_registry(get_settings())
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)

app.include_router(generate_router)
app.include_router(extract_router)


def _get_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)