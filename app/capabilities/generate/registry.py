"""
Backend registry (Section 4): maps a backend name string to a live adapter
instance, built once at startup from config.

This is what makes "add a backend" mean "write one adapter class, add one
line here" instead of touching route handlers. main.py and router.py only
ever ask the registry for a backend by name - they never construct an
adapter directly.
"""

from app.capabilities.generate.base import GenerationBackend
from app.capabilities.generate.backends.gemini import GeminiGenerationBackend
from app.capabilities.generate.backends.groq import GroqGenerationBackend
from app.capabilities.generate.backends.ollama_local import OllamaGenerationBackend

from app.config import Settings


class GenerationRegistry:
    """Holds every configured generation backend, keyed by name."""

    def __init__(self, backends: dict[str, GenerationBackend], primary: str) -> None:
        self._backends = backends
        self.primary = primary

    def get(self, name: str | None) -> GenerationBackend:
        """Resolve a backend by name, falling back to the configured
        primary if the caller didn't pin one."""
        resolved = name or self.primary
        if resolved not in self._backends:
            raise KeyError(
                f"Unknown generation backend '{resolved}'. "
                f"Known backends: {list(self._backends)}"
            )
        return self._backends[resolved]

    def all(self) -> list[GenerationBackend]:
        """All configured backends - used by /v1/health to check every one."""
        return list(self._backends.values())

    def names(self) -> list[str]:
        """All configured backend names - used by router.py (G2) to
        validate a caller-pinned backend before building a fallback order,
        and by observability/service.py's /v1/models."""
        return list(self._backends)


def build_generation_registry(settings: Settings) -> GenerationRegistry:
    """Construct every generation adapter from settings.

    G1 will add two more lines here (Groq, Qwen-local) and nothing else
    in this file changes shape - that's the pattern working as intended.
    """
    backends: dict[str, GenerationBackend] = {
        "gemini": GeminiGenerationBackend(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model_name,
        ),
        "groq": GroqGenerationBackend(
            api_key=settings.groq_api_key,
            model_name=settings.groq_model_name,
        ),
        "qwen_local": OllamaGenerationBackend(
            host=settings.ollama_host,
            model_name=settings.ollama_model_name,
        ),
    }
    return GenerationRegistry(backends=backends, primary=settings.generation_primary_backend)