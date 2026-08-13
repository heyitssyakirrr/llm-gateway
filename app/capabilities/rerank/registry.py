"""
Rerank backend registry (Section 4). Same shape as generate/registry.py
and embed/registry.py, deliberately - even though G4 only configures one
backend today, keeping the registry pattern identical means adding a
second rerank provider later really is "write one adapter, add one
line here," not a special case because this capability started
single-backend.
"""

from app.capabilities.rerank.backends.cohere import CohereRerankBackend
from app.capabilities.rerank.base import RerankBackend
from app.config import Settings


class RerankRegistry:
    """Holds every configured rerank backend, keyed by name."""

    def __init__(self, backends: dict[str, RerankBackend], primary: str) -> None:
        self._backends = backends
        self.primary = primary

    def get(self, name: str | None) -> RerankBackend:
        """Resolve a backend by name, falling back to the configured
        primary if the caller didn't pin one."""
        resolved = name or self.primary
        if resolved not in self._backends:
            raise KeyError(
                f"Unknown rerank backend '{resolved}'. "
                f"Known backends: {list(self._backends)}"
            )
        return self._backends[resolved]

    def all(self) -> list[RerankBackend]:
        """All configured backends - used by /v1/health to check every one."""
        return list(self._backends.values())

    def names(self) -> list[str]:
        """All configured backend names - used by router.py to validate a
        caller-pinned backend before building a fallback order."""
        return list(self._backends)


def build_rerank_registry(settings: Settings) -> RerankRegistry:
    """Construct every rerank adapter from settings."""
    backends: dict[str, RerankBackend] = {
        "cohere": CohereRerankBackend(
            api_key=settings.cohere_api_key,
            model_name=settings.cohere_rerank_model_name,
        ),
    }
    return RerankRegistry(backends=backends, primary=settings.rerank_primary_backend)
