import os

from agents.agent import Agent
from agents.guardrails import PriceGuardrailError, relative_spread, validate_price
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.neural_network_agent import NeuralNetworkAgent
from agents.preprocessor import Preprocessor


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

    def price(self, description: str) -> float:
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
        estimates = [specialist, frontier, neural_network]
        spread = relative_spread(estimates)
        if spread > self.MAX_RELATIVE_SPREAD:
            raise PriceGuardrailError(
                "pricing models disagree too strongly "
                f"(relative spread {spread:.2f} > {self.MAX_RELATIVE_SPREAD:.2f})"
            )
        combined = validate_price(
            frontier * 0.8 + specialist * 0.1 + neural_network * 0.1
        )
        self.log(f"Ensemble Agent complete - returning ${combined:.2f}")
        return combined
