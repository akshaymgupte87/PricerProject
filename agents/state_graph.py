"""Resumable deal workflow with optional native LangGraph execution."""

from __future__ import annotations

import uuid
from typing import Any, TypedDict

from agents.deals import Deal, Opportunity
from agents.ensemble_agent import EnsembleAgent
from pricer.intelligence_store import IntelligenceStore


class DealGraphState(TypedDict, total=False):
    run_id: str
    deal: dict[str, Any]
    opportunity: dict[str, Any]
    status: str
    error: str
    attempts: int


class DealStateGraph:
    """Price, checkpoint, pause for approval, and resume a single deal."""

    def __init__(
        self,
        ensemble: EnsembleAgent,
        store: IntelligenceStore,
        *,
        approval_confidence: float = 0.85,
        max_attempts: int = 2,
    ):
        self.ensemble = ensemble
        self.store = store
        self.approval_confidence = approval_confidence
        self.max_attempts = max_attempts
        self.graph = self._build_langgraph()

    def _price(self, state: DealGraphState) -> DealGraphState:
        deal = Deal.model_validate(state["deal"])
        attempts = state.get("attempts", 0) + 1
        try:
            estimate = self.ensemble.estimate(deal.product_description)
            opportunity = Opportunity(
                deal=deal,
                estimate=estimate.value,
                discount=estimate.value - deal.price,
                model_estimates=estimate.model_estimates,
                confidence=estimate.confidence,
                confidence_low=estimate.confidence_low,
                confidence_high=estimate.confidence_high,
                evidence=estimate.evidence,
                explanation=estimate.explanation,
                status="pending",
            )
            return {**state, "opportunity": opportunity.model_dump(), "attempts": attempts}
        except Exception as error:
            return {**state, "error": f"{type(error).__name__}: {error}", "attempts": attempts}

    def _route_after_price(self, state: DealGraphState) -> str:
        if state.get("error") and state.get("attempts", 0) < self.max_attempts:
            return "retry"
        if state.get("error"):
            return "failed"
        return "persist"

    def _persist(self, state: DealGraphState) -> DealGraphState:
        opportunity = Opportunity.model_validate(state["opportunity"])
        self.store.upsert_opportunity(opportunity)
        status = "approval_required"
        if opportunity.confidence >= self.approval_confidence:
            status = "approval_recommended"
        updated = {**state, "status": status}
        self.store.save_checkpoint(state["run_id"], updated)
        return updated

    @staticmethod
    def _failed(state: DealGraphState) -> DealGraphState:
        return {**state, "status": "failed"}

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None
        builder = StateGraph(DealGraphState)
        builder.add_node("price", self._price)
        builder.add_node("persist", self._persist)
        builder.add_node("failed", self._failed)
        builder.add_edge(START, "price")
        builder.add_conditional_edges(
            "price",
            self._route_after_price,
            {"retry": "price", "persist": "persist", "failed": "failed"},
        )
        builder.add_edge("persist", END)
        builder.add_edge("failed", END)
        return builder.compile()

    def run(self, deal: Deal, run_id: str | None = None) -> DealGraphState:
        initial: DealGraphState = {
            "run_id": run_id or uuid.uuid4().hex,
            "deal": deal.model_dump(),
            "attempts": 0,
            "status": "running",
        }
        if self.graph is not None:
            result = self.graph.invoke(initial)
        else:
            result = self._price(initial)
            while self._route_after_price(result) == "retry":
                result = self._price(result)
            result = self._persist(result) if "opportunity" in result else self._failed(result)
        self.store.save_checkpoint(result["run_id"], result)
        return result

    def resume_approval(
        self,
        run_id: str,
        decision: str,
        *,
        corrected_price: float | None = None,
        reason: str = "",
    ) -> DealGraphState:
        state = self.store.load_checkpoint(run_id)
        if not state or "opportunity" not in state:
            raise KeyError(f"unknown resumable run: {run_id}")
        opportunity = Opportunity.model_validate(state["opportunity"])
        updated = self.store.record_feedback(
            opportunity.deal.url,
            decision,
            corrected_price=corrected_price,
            reason=reason,
        )
        state["opportunity"] = updated.model_dump()
        state["status"] = decision
        self.store.save_checkpoint(run_id, state)
        return state
