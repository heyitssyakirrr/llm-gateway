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
from app.capabilities.embed.registry import EmbeddingRegistry, build_embedding_registry
from app.capabilities.embed.routes import router as embed_router
from app.capabilities.generate.registry import GenerationRegistry, build_generation_registry
from app.capabilities.generate.routes import router as generate_router
from app.capabilities.extract.routes import router as extract_router
from app.config import get_settings
from app.logging_db import init_db
from app.observability.routes import router as observability_router
from app.schemas.health import BackendHealth, HealthResponse


# things that happens once, when the app starts up, not once per request.  
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings = get_settings()
    app.state.generation_registry = build_generation_registry(settings)
    app.state.embedding_registry = build_embedding_registry(settings)
    yield


app = FastAPI(title="LLM Gateway", lifespan=lifespan)

app.include_router(generate_router)
app.include_router(extract_router)
app.include_router(embed_router)
app.include_router(observability_router)


def _get_generation_registry(request: Request) -> GenerationRegistry:
    return request.app.state.generation_registry


def _get_embedding_registry(request: Request) -> EmbeddingRegistry:
    return request.app.state.embedding_registry


@app.get("/v1/health", response_model=HealthResponse)
async def health(
    http_request: Request,
    caller_id: str = Depends(verify_api_key),
) -> HealthResponse:
    generation_registry = _get_generation_registry(http_request)
    embedding_registry = _get_embedding_registry(http_request)

    # Both registries' health checks run concurrently, not sequentially -
    # same reasoning as observability/service.py's build_models_response:
    # this endpoint shouldn't get slower every time a capability gains a
    # new backend.
    generation_backends = generation_registry.all()
    embedding_backends = embedding_registry.all()
    generation_results, embedding_results = await asyncio.gather(
        asyncio.gather(*(b.health_check() for b in generation_backends)),
        asyncio.gather(*(b.health_check() for b in embedding_backends)),
    )

    backends = [
        BackendHealth(capability="generate", backend=r.backend, reachable=r.reachable, detail=r.detail)
        for r in generation_results
    ] + [
        BackendHealth(capability="embed", backend=r.backend, reachable=r.reachable, detail=r.detail)
        for r in embedding_results
    ]
    return HealthResponse(backends=backends)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)