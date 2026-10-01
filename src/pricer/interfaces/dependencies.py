"""Authentication and lazy application dependencies."""
import os
from functools import lru_cache
from fastapi import Header, HTTPException
from pricer.config import load_environment


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    load_environment()
    configured = os.getenv("PRICER_API_KEY", "").strip()
    if configured and x_api_key != configured:
        raise HTTPException(status_code=401, detail="invalid API key")


@lru_cache(maxsize=1)
def framework():
    from pricer.application.framework import DealAgentFramework
    return DealAgentFramework()
