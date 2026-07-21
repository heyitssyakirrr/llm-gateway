"""
Embedding backend registry (Section 4). Identical shape to
`capabilities/generate/registry.py` - maps a backend name string to a
live adapter instance, built once at startup from config. Adding a
backend later is "write one adapter class, add one line here", same
promise as generation's registry.
"""

from app.capabilities.embed.backends.cohere import CohereEmbeddingBackend
from app.capabilities.embed.backends.gemini import GeminiEmbeddingBackend
from app.capabilities.embed.base import EmbeddingBackend
from app.config import Settings


class EmbeddingRegistry:
    """Holds every configured embedding backend, keyed by name."""

    def __init__(self, backends: dict[str, EmbeddingBackend], primary: str) -> None:
        self._backends = backends
        self.primary = primary

    def get(self, name: str | None) -> EmbeddingBackend:
        """Resolve a backend by name, falling back to the configured
        primary if the caller didn't pin one."""
        resolved = name or self.primary
        if resolved not in self._backends:
            raise KeyError(
                f"Unknown embedding backend '{resolved}'. "
                f"Known backends: {list(self._backends)}"
            )
        return self._backends[resolved]

    def all(self) -> list[EmbeddingBackend]:
        """All configured backends - used by /v1/health to check every one."""
        return list(self._backends.values())

    def names(self) -> list[str]:
        """All configured backend names - used by router.py to validate a
        caller-pinned backend before building a fallback order."""
        return list(self._backends)


def build_embedding_registry(settings: Settings) -> EmbeddingRegistry:
    """Construct every embedding adapter from settings."""
    backends: dict[str, EmbeddingBackend] = {
        "gemini": GeminiEmbeddingBackend(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embedding_model_name,
        ),
        "cohere": CohereEmbeddingBackend(
            api_key=settings.cohere_api_key,
            model_name=settings.cohere_embedding_model_name,
        ),
    }
    return EmbeddingRegistry(backends=backends, primary=settings.embedding_primary_backend)
