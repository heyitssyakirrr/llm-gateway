"""
X-API-Key authentication dependency.

Deliberately the simplest possible version of auth (Section 3.6): a
static shared secret per known caller, checked on every route via
`Depends(verify_api_key)`. No OAuth, no rotation, no scopes - the goal is
just "identify the caller, reject anyone without a valid key."
"""

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validates X-API-Key and returns the resolved
    caller_id for use in logging (Section 5's `caller_id` field).

    Raises 401 if the header is missing or doesn't match a configured key.
    Applied to every route from G0 onward - see Section 3.6's rationale for
    why this isn't deferred until a second consumer shows up.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    keys = get_settings().api_keys_by_key()
    caller_id = keys.get(x_api_key)
    if caller_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return caller_id
