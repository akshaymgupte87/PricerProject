"""HTTP routes delegate pricing and persistence to the application service."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pricer.agents.deals import Deal
from pricer.interfaces.dependencies import framework, require_api_key
from pricer.interfaces.schemas import EstimateRequest, FeedbackRequest, SavedSearchRequest

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pricer-intelligence"}


@router.get("/opportunities", dependencies=[Depends(require_api_key)])
def opportunities(status: str | None = None, service=Depends(framework)) -> list[dict]:
    return [item.model_dump() for item in service.store.list_opportunities(status)]


@router.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(request: FeedbackRequest, service=Depends(framework)) -> dict:
    try:
        return service.submit_feedback(
            request.url,
            request.decision,
            corrected_price=request.corrected_price,
            reason=request.reason,
        ).model_dump()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.get("/price-history", dependencies=[Depends(require_api_key)])
def price_history(url: str, service=Depends(framework)) -> list[dict]:
    return service.price_history(url)


@router.get("/price-insights", dependencies=[Depends(require_api_key)])
def price_insights(url: str, service=Depends(framework)) -> dict:
    return service.store.price_insights(url)


@router.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics(service=Depends(framework)) -> list[dict]:
    return service.store.metric_summary()


@router.get("/evaluation", dependencies=[Depends(require_api_key)])
def evaluation(service=Depends(framework)) -> dict:
    return service.store.evaluation_summary()


@router.get("/saved-searches", dependencies=[Depends(require_api_key)])
def saved_searches(service=Depends(framework)) -> list[dict]:
    return service.store.list_saved_searches()


@router.post("/saved-searches", dependencies=[Depends(require_api_key)])
def save_search(request: SavedSearchRequest, service=Depends(framework)) -> dict[str, int | str]:
    identifier = service.store.save_search(request.name, request.query, request.filters)
    return {"id": identifier, "status": "saved"}


@router.get("/saved-searches/{identifier}/matches", dependencies=[Depends(require_api_key)])
def saved_search_matches(identifier: int, service=Depends(framework)) -> list[dict]:
    try:
        return [item.model_dump() for item in service.store.saved_search_matches(identifier)]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/estimate", dependencies=[Depends(require_api_key)])
def estimate(request: EstimateRequest, service=Depends(framework)) -> dict:
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


@router.post("/scan", status_code=202, dependencies=[Depends(require_api_key)])
def scan(background_tasks: BackgroundTasks, service=Depends(framework)) -> dict[str, str]:
    background_tasks.add_task(service.run)
    return {"status": "accepted"}
