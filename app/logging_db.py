"""
Structured request logging - the `request_log` table from Section 5.

This is, per your own progress notes, "the single most learning-dense
artifact in this project": every field exists to answer a specific
"why did this happen" question later, via /v1/stats (built in G2).

We write the FULL schema starting at G0 - even though fields like
`fallback_chain` and `error_type` won't have interesting values until G1/G2
exist (no failover logic yet, so `fallback_chain` is always `[]`). Adding
a column later would mean an ALTER TABLE + backfilling old rows with
nulls; defining the full shape now avoids that.

Implementation note: sqlite3 is synchronous. For a personal-project
request volume, one blocking insert per request is not a real bottleneck,
but it DOES block the event loop for that moment - worth knowing as a
conscious tradeoff, not an oversight. If this ever becomes a real
bottleneck, the fix is `asyncio.to_thread(log_request, ...)` or an actual
async sqlite driver (aiosqlite) - noted as backlog, not needed at G0.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    caller_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    backend_requested TEXT,
    backend_used TEXT,
    model_name TEXT,
    params_used TEXT,
    fallback_chain TEXT,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_estimate REAL,
    retries INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL,
    error_type TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def _connection():
    conn = sqlite3.connect(get_settings().db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the request_log table if it doesn't exist. Call once at
    app startup (see main.py's lifespan)."""
    with _connection() as conn:
        conn.execute(_SCHEMA)


@dataclass
class RequestLogEntry:
    """Everything one row needs. Kept as a dataclass (not raw kwargs) so
    callers get type-checked field names instead of typo-prone strings."""

    request_id: str
    caller_id: str
    capability: str
    endpoint: str
    backend_used: str | None
    success: bool
    backend_requested: str | None = None
    model_name: str | None = None
    params_used: dict = field(default_factory=dict)
    fallback_chain: list[str] = field(default_factory=list)
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_estimate: float = 0.0
    retries: int = 0
    error_type: str | None = None


def log_request(entry: RequestLogEntry) -> None:
    """Write one row to request_log.

    IMPORTANT (Section 7): `params_used` must only ever contain call
    metadata (temperature, max_tokens, ...) - never the prompt or response
    text itself. Enforce that at the call site, not here, since this
    function has no way to know what's "content" vs "metadata" in an
    arbitrary dict.
    """
    with _connection() as conn:
        conn.execute(
            """
            INSERT INTO request_log (
                request_id, caller_id, capability, endpoint,
                backend_requested, backend_used, model_name,
                params_used, fallback_chain, latency_ms,
                prompt_tokens, completion_tokens, cost_estimate,
                retries, success, error_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.request_id,
                entry.caller_id,
                entry.capability,
                entry.endpoint,
                entry.backend_requested,
                entry.backend_used,
                entry.model_name,
                json.dumps(entry.params_used),
                json.dumps(entry.fallback_chain),
                entry.latency_ms,
                entry.prompt_tokens,
                entry.completion_tokens,
                entry.cost_estimate,
                entry.retries,
                int(entry.success),
                entry.error_type,
                datetime.now(UTC).isoformat(),
            ),
        )
