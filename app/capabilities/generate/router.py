"""
Router (Section 4): picks which backend actually serves a request.

At G0 this is deliberately thin - it just resolves the requested (or
primary) backend via the registry. There is no failover yet; that's G2's
job (Section 3.6's RPM/TPM-vs-RPD distinction). We return a
`RoutingDecision` with an (empty, for now) `fallback_chain` so that when
G2 adds real failover, main.py's calling code doesn't need to change shape
- only this function's internals grow.
"""

from dataclasses import dataclass, field

from app.capabilitites.generate.base import GenerationBackend
from app.capabilitites.generate.registry import GenerationRegistry


@dataclass
class RoutingDecision:
    backend: GenerationBackend
    fallback_chain: list[str] = field(default_factory=list)


def route_generation(registry: GenerationRegistry, requested_backend: str | None) -> RoutingDecision:
    """Resolve which backend serves a /v1/generate call.

    G2 will change this to: try the requested/primary backend, and on a
    RateLimitedError retry with backoff, or on a QuotaExceededError move
    to the next backend in a configured order, appending each attempted
    name to `fallback_chain` along the way.
    """
    backend = registry.get(requested_backend)
    return RoutingDecision(backend=backend, fallback_chain=[])
