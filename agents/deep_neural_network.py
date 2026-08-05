import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
import logging


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.device = None
        self.y_std = Y_STD
        self.y_mean = Y_MEAN

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self, model_config=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        # New checkpoints include their architecture, so load() normally builds the model.
        # Supplying a config here remains useful when constructing an untrained network.
        if model_config is not None:
            self._build_model(model_config)

    def _build_model(self, model_config, vectorizer_config=None):
        config = {
            "input_size": int(model_config.get("input_size", 5000)),
            "num_layers": int(model_config.get("num_layers", 10)),
            "hidden_size": int(model_config.get("hidden_size", 4096)),
            "dropout_prob": float(model_config.get("dropout_prob", 0.2)),
        }
        if vectorizer_config is None:
            # Backward-compatible settings for original and legacy checkpoints.
            vectorizer_config = {
                "n_features": config["input_size"],
                "stop_words": "english",
                "binary": True,
            }
        self.vectorizer = HashingVectorizer(**vectorizer_config)
        self.model = DeepNeuralNetwork(**config).to(self.device)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Locally trained checkpoints are self-describing. Also accept the original
        # weights-only .pth format for backwards compatibility.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self._build_model(
                checkpoint["model_config"], checkpoint.get("vectorizer_config")
            )
            self.y_mean = float(checkpoint["target_mean"])
            self.y_std = float(checkpoint["target_std"])
            state_dict = checkpoint["model_state_dict"]
        else:
            self._build_model(
                {"input_size": 5000, "num_layers": 10, "hidden_size": 4096, "dropout_prob": 0.2}
            )
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def inference(self, text):
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Call setup() and load() with a trained checkpoint before inference.")
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * self.y_std + self.y_mean) - 1
            result = result.item()
        return max(0, result)
