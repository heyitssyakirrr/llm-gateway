"""
Shared health-check result shape.

Every capability's adapters (generate, embed, and rerank later) report
reachability the same way, so this is one dataclass, not one per
capability. See `common/errors.py` for the same reasoning applied to the
error taxonomy.
"""

from dataclasses import dataclass


@dataclass
class HealthStatus:
    """Result of a single backend's health check (Section 3.8)."""

    backend: str
    reachable: bool
    detail: str = ""
