"""
Backend registry (Section 4): maps a backend name string to a live adapter
instance, built once at startup from config.

This is what makes "add a backend" mean "write one adapter class, add one
line here" instead of touching route handlers. main.py and router.py only
ever ask the registry for a backend by name - they never construct an
adapter directly.
"""

from app.backends.base import GenerationBackend
from app.backends.gemini_generate import GeminiGenerationBackend
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
    }
    return GenerationRegistry(backends=backends, primary=settings.generation_primary_backend)
