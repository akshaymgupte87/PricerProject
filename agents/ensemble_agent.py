import os

from pydantic import BaseModel, Field

from agents.agent import Agent
from agents.guardrails import PriceGuardrailError, relative_spread, validate_price
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


class PriceEstimate(BaseModel):
    value: float
    model_estimates: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_low: float
    confidence_high: float
    evidence: list[dict] = Field(default_factory=list)
    explanation: str


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW
    MAX_RELATIVE_SPREAD = float(os.getenv("PRICER_MAX_MODEL_SPREAD_RATIO", "2.0"))

    def __init__(self, collection):
        """
        Create an instance of Ensemble, by creating each of the models
        And loading the weights of the Ensemble
        """
        self.log("Initializing Ensemble Agent")
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.neural_network = NeuralNetworkAgent()
        self.preprocessor = Preprocessor()
        self.log("Ensemble Agent is ready")

    def estimate(self, description: str) -> PriceEstimate:
        """
        Run this ensemble model
        Ask each of the models to price the product
        Then use the Linear Regression model to return the weighted price
        :param description: the description of a product
        :return: an estimate of its price
        """
        self.log("Running Ensemble Agent - preprocessing text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Pre-processed text using {self.preprocessor.model_name}")
        specialist = validate_price(self.specialist.price(rewrite))
        frontier = validate_price(self.frontier.price(rewrite))
        neural_network = validate_price(self.neural_network.price(rewrite))
        model_estimates = {
            "specialist": specialist,
            "frontier_rag": frontier,
            "neural_network": neural_network,
        }
        estimates = list(model_estimates.values())
        spread = relative_spread(estimates)
        if spread > self.MAX_RELATIVE_SPREAD:
            raise PriceGuardrailError(
                "pricing models disagree too strongly "
                f"(relative spread {spread:.2f} > {self.MAX_RELATIVE_SPREAD:.2f})"
            )
        combined = validate_price(
            frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
        )
        raw_evidence = self.frontier.evidence() if hasattr(self.frontier, "evidence") else []
        evidence = raw_evidence if isinstance(raw_evidence, list) else []
        agreement = max(0.0, 1.0 - spread / max(self.MAX_RELATIVE_SPREAD, 0.01))
        evidence_factor = min(len(evidence) / 5, 1.0)
        confidence = min(0.99, max(0.05, 0.75 * agreement + 0.25 * evidence_factor))
        half_width = max(combined * (0.08 + 0.35 * (1.0 - confidence)), 10.0)
        explanation = (
            f"Weighted estimate from three models (${min(estimates):,.0f}–"
            f"${max(estimates):,.0f}); {len(evidence)} retrieved comparables; "
            f"model agreement {agreement:.0%}."
        )
        self.log(
            f"Ensemble Agent complete - returning ${combined:.2f} "
            f"with {confidence:.0%} confidence"
        )
        return PriceEstimate(
            value=combined,
            model_estimates=model_estimates,
            confidence=confidence,
            confidence_low=max(0.01, combined - half_width),
            confidence_high=combined + half_width,
            evidence=evidence,
            explanation=explanation,
        )

    def price(self, description: str) -> float:
        """Backward-compatible scalar pricing interface."""
        return self.estimate(description).value
