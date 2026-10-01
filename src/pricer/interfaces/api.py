"""FastAPI application factory. Importing it never initializes pricing models."""
from fastapi import FastAPI
from pricer.interfaces.dependencies import framework, require_api_key
from pricer.interfaces.routes import router
from pricer.interfaces.schemas import EstimateRequest, FeedbackRequest, SavedSearchRequest


def create_app() -> FastAPI:
    application = FastAPI(
        title="Pricer Intelligence API",
        version="0.2.0",
        description="Local-first pricing, evidence, approvals, history, and monitoring.",
    )
    application.include_router(router)
    return application


app = create_app()
