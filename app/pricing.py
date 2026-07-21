"""
Pricing table for cost_estimate (Section 3.7).

Every backend/model in use today is free, but we compute cost_estimate for
real - $0.00 entered explicitly - so /v1/stats stays meaningful the moment
a paid backend (e.g. Claude, GPT-4o-mini) is ever added, with zero changes
to the logging/stats code path. Only this table needs a new row.

Prices are per 1,000 tokens, input/output split where a provider prices
them differently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_1k: float
    output_per_1k: float


# (backend, model_name) -> Price
PRICING_TABLE: dict[tuple[str, str], Price] = {
    ("gemini", "gemini-2.5-flash"): Price(input_per_1k=0.0, output_per_1k=0.0),
    # Embedding calls have no "completion" side - output_per_1k is 0.0 for
    # both rows not because these happen to be free, but because that's
    # structurally what an embedding call is (input tokens only).
    ("gemini", "gemini-embedding-001"): Price(input_per_1k=0.0, output_per_1k=0.0),
    ("cohere", "embed-english-v3.0"): Price(input_per_1k=0.0, output_per_1k=0.0),
}


def estimate_cost(
    backend: str,
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float:
    """Compute cost_estimate from the pricing table.

    Returns 0.0 (with no error) for an unknown (backend, model) pair rather
    than raising - a missing pricing row shouldn't ever break a request,
    it should just mean $0.00 gets logged until someone adds the row.
    """
    price = PRICING_TABLE.get((backend, model_name))
    if price is None:
        return 0.0
    prompt_cost = (prompt_tokens or 0) / 1000 * price.input_per_1k
    completion_cost = (completion_tokens or 0) / 1000 * price.output_per_1k
    return round(prompt_cost + completion_cost, 8)
