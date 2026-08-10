"""Secured FastAPI surface for the local deal-intelligence system."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agents.deals import Deal
from deal_agent_framework import DealAgentFramework


app = FastAPI(
    title="Pricer Intelligence API",
    version="0.2.0",
    description="Local-first pricing, evidence, approvals, history, and monitoring.",
)


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


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    configured = os.getenv("PRICER_API_KEY", "").strip()
    if configured and x_api_key != configured:
        raise HTTPException(status_code=401, detail="invalid API key")


@lru_cache(maxsize=1)
def framework() -> DealAgentFramework:
    return DealAgentFramework()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pricer-intelligence"}


@app.get("/opportunities", dependencies=[Depends(require_api_key)])
def opportunities(status: str | None = None) -> list[dict]:
    return [item.model_dump() for item in framework().store.list_opportunities(status)]


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(request: FeedbackRequest) -> dict:
    try:
        return framework().submit_feedback(
            request.url,
            request.decision,
            corrected_price=request.corrected_price,
            reason=request.reason,
        ).model_dump()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/price-history", dependencies=[Depends(require_api_key)])
def price_history(url: str) -> list[dict]:
    return framework().price_history(url)


@app.get("/price-insights", dependencies=[Depends(require_api_key)])
def price_insights(url: str) -> dict:
    return framework().store.price_insights(url)


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics() -> list[dict]:
    return framework().store.metric_summary()


@app.get("/evaluation", dependencies=[Depends(require_api_key)])
def evaluation() -> dict:
    return framework().store.evaluation_summary()


@app.get("/saved-searches", dependencies=[Depends(require_api_key)])
def saved_searches() -> list[dict]:
    return framework().store.list_saved_searches()


@app.post("/saved-searches", dependencies=[Depends(require_api_key)])
def save_search(request: SavedSearchRequest) -> dict[str, int | str]:
    identifier = framework().store.save_search(request.name, request.query, request.filters)
    return {"id": identifier, "status": "saved"}


@app.get("/saved-searches/{identifier}/matches", dependencies=[Depends(require_api_key)])
def saved_search_matches(identifier: int) -> list[dict]:
    try:
        return [item.model_dump() for item in framework().store.saved_search_matches(identifier)]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/estimate", dependencies=[Depends(require_api_key)])
def estimate(request: EstimateRequest) -> dict:
    service = framework()
    service.init_agents_as_needed()
    deal = Deal(
        product_description=request.description,
        price=request.asking_price,
        url=request.url,
    )
    with service.observability.trace("manual_estimate"):
        opportunity = service.planner.run(deal)
        service.store.upsert_opportunity(opportunity)
    return opportunity.model_dump()


@app.post("/scan", status_code=202, dependencies=[Depends(require_api_key)])
def scan(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(framework().run)
    return {"status": "accepted"}
