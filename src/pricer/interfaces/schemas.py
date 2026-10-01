"""Validated HTTP request schemas."""
from typing import Any
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    url: str
    decision: str
    corrected_price: float | None = Field(default=None, gt=0)
    reason: str = ""


class SavedSearchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=500)
    filters: dict[str, Any] = Field(default_factory=dict)


class EstimateRequest(BaseModel):
    description: str = Field(min_length=3, max_length=2_000)
    asking_price: float = Field(gt=0)
    url: str = "https://local.invalid/manual-estimate"
