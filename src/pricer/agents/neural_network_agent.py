from pricer.agents.agent import Agent
from pricer.agents.deep_neural_network import DeepNeuralNetworkInference
from pathlib import Path
from pricer.config import Settings, runtime_home


PROJECT_ROOT = runtime_home()
DEFAULT_CHECKPOINT = Settings.from_env().checkpoint_path


class NeuralNetworkAgent(Agent):
    name = "Neural Network Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("Neural Network Agent is initializing")
        checkpoint = Settings.from_env().checkpoint_path
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} does not exist. Build it locally with "
                r"uv run pricer train --epochs 5"
            )
        self.neural_network = DeepNeuralNetworkInference()
        self.neural_network.setup()
        self.neural_network.load(checkpoint)
        self.log("Neural Network Agent is ready and weights are loaded")

    def price(self, description: str) -> float:
        """
        Use the Deep Neural Network to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float
        """
        self.log("Neural Network Agent is starting a prediction")
        result = self.neural_network.inference(description)
        self.log(f"Neural Network Agent completed - predicting ${result:.2f}")
        return result
