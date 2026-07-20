"""
Central configuration, loaded from environment variables.

Every secret (API keys) and every provider-specific detail (model names)
lives here, loaded from env - never hardcoded in route handlers or
adapters (Section 7.5's Security checklist). Load once per process via
`get_settings()` (cached), so we're not re-parsing env vars per request.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # reads a local .env file in dev; no-op in prod if absent


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