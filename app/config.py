"""
Central configuration, loaded from environment variables.
Load once per process via `get_settings()` (cached), so we're 
not re-parsing env vars per request.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # --- Provider credentials ---
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))

    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))

    ollama_host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )

    # --- Provider model choice (never hardcode elsewhere - free tiers shift) ---
    gemini_model_name: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL_NAME", "gemini-flash-latest")
    )
    
    groq_model_name: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    )

    ollama_model_name: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL_NAME", "qwen2.5:3b-instruct")
    )

    # --- Which backend serves a capability when the caller doesn't pin one ---
    generation_primary_backend: str = field(
        default_factory=lambda: os.environ.get("GENERATION_PRIMARY_BACKEND", "gemini")
    )

    # --- Resilience (backoff + failover) ---
    # Order backends are tried in when the primary/pinned one fails in a
    # way that isn't fixed by retrying it. The requested/primary backend
    # is always tried first regardless of its position here
    # see router.py's _build_attempt_order.
    generation_fallback_order: list[str] = field(
        default_factory=lambda: [
            name.strip()
            for name in os.environ.get(
                "GENERATION_FALLBACK_ORDER", "gemini,groq,qwen_local"
            ).split(",")
            if name.strip()
        ]
    )

    # Max backoff-and-retry attempts on a SINGLE backend for a
    # RateLimitedError before giving up on it and moving to the next
    # backend in generation_fallback_order. 0 disables retries entirely
    # (first RateLimitedError fails over immediately).
    generation_max_retries_per_backend: int = field(
        default_factory=lambda: int(os.environ.get("GENERATION_MAX_RETRIES_PER_BACKEND", "3"))
    )

    # Backoff base delay, in seconds, for the exponential-backoff-with-
    # full-jitter formula in resilience.py: actual delay for attempt N is
    # a random value in [0, min(max, base * 2**N)).
    generation_backoff_base_seconds: float = field(
        default_factory=lambda: float(os.environ.get("GENERATION_BACKOFF_BASE_SECONDS", "0.5"))
    )

    # Backoff delay cap, in seconds - keeps retries from ever waiting an
    # unreasonably long time on a single backend before failing over.
    generation_backoff_max_seconds: float = field(
        default_factory=lambda: float(os.environ.get("GENERATION_BACKOFF_MAX_SECONDS", "8.0"))
    )

    # --- Auth: "caller_name:key,caller_name2:key2" ---
    raw_api_keys: str = field(default_factory=lambda: os.environ.get("GATEWAY_API_KEYS", ""))

    # --- Local SQLite log store ---
    db_path: str = field(default_factory=lambda: os.environ.get("GATEWAY_DB_PATH", "gateway.db"))

    def api_keys_by_key(self) -> dict[str, str]:
        """Parse GATEWAY_API_KEYS into {key: caller_name} for O(1) lookup on
        every request. Parsed lazily rather than at import time so tests can
        swap env vars before calling this."""
        result: dict[str, str] = {}
        for pair in self.raw_api_keys.split(","):       # split into ["syakir:dev-secret-123"] if there is more than one caller
            pair = pair.strip()                         # eg: {"dev-secret-123": "syakir", "9f3a7c21b8e04d5e": "policy-rag"}
            if not pair:
                continue
            caller_name, _, key = pair.partition(":")   # "syakir", ":", "dev-secret-123"
            if caller_name and key:
                result[key] = caller_name               # {"dev-secret-123": "syakir"}
        return result


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton - env vars are read once per process."""
    return Settings()