"""Original eight-layer bag-of-words experiment, with batch-sized dense arrays.

This research model is separate from the production checkpoint trainer in
train_deep_neural_network.py; their architectures and targets differ.
"""

import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split

from .baselines import TextPricer


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        sizes = [input_size, 128, 64, 64, 64, 64, 64, 64, 1]
        for index, (left, right) in enumerate(zip(sizes, sizes[1:]), 1):
            setattr(self, f"layer{index}", nn.Linear(left, right))
        self.relu = nn.ReLU()

    def forward(self, value):
        for index in range(1, 8):
            value = self.relu(getattr(self, f"layer{index}")(value))
        return self.layer8(value)

    def predict(self, features):
        self.eval()
        with torch.no_grad():
            return self(torch.from_numpy(features.toarray().astype(np.float32))).numpy().ravel()


def fit_neural_network(train, epochs=2, batch_size=64, seed=42):
    """Preserve the 1% internal validation split and Adam/MSE training recipe."""
    if len(train) < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("Need at least two items, one epoch, and a positive batch size")
    vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
    features = vectorizer.transform([item.summary or "" for item in train])
    targets = np.array([item.price for item in train], dtype=np.float32)
    train_ids, val_ids = train_test_split(np.arange(len(train)), test_size=0.01, random_state=seed)
    # Avoid changing a caller's torch RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NeuralNetwork(features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        history = []
        for epoch in range(epochs):
            model.train()
            order = train_ids[torch.randperm(len(train_ids)).numpy()]
            training_loss = 0.0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                batch = torch.from_numpy(features[indices].toarray().astype(np.float32))
                target = torch.from_numpy(targets[indices]).unsqueeze(1)
                optimizer.zero_grad()
                loss = loss_function(model(batch), target)
                loss.backward()
                optimizer.step()
                training_loss += loss.item() * len(indices)
            validation_loss = 0.0
            for start in range(0, len(val_ids), batch_size):
                indices = val_ids[start:start + batch_size]
                predictions = model.predict(features[indices])
                validation_loss += float(np.sum((predictions - targets[indices]) ** 2))
            metrics = {"epoch": epoch + 1, "train_mse": training_loss / len(train_ids), "validation_mse": validation_loss / len(val_ids)}
            history.append(metrics)
            print(metrics, flush=True)
    return TextPricer(model, vectorizer), history
