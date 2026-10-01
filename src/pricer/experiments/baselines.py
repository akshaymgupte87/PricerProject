"""Fitted baseline predictors keep their own model and vectorizer state."""

from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = ["weight", "weight_unknown", "text_length"]


def get_features(item):
    weight = item.weight or 0
    return {"weight": weight, "weight_unknown": int(weight == 0), "text_length": len(item.summary or "")}


def list_to_dataframe(items):
    frame = pd.DataFrame([get_features(item) for item in items], columns=FEATURE_COLUMNS)
    frame["price"] = [item.price for item in items]
    return frame


def random_pricer(seed=42):
    rng = random.Random(seed)
    def predict(item):
        return rng.randrange(1, 1000)
    return predict


def constant_pricer(train):
    average = sum(item.price for item in train) / len(train)
    def predict(item):
        return average
    return predict


@dataclass
class FeaturePricer:
    model: object

    @property
    def __name__(self):
        return "feature_linear_regression"

    def __call__(self, item):
        frame = pd.DataFrame([get_features(item)], columns=FEATURE_COLUMNS)
        return float(self.model.predict(frame)[0])


def fit_feature_regression(train):
    frame = list_to_dataframe(train)
    return FeaturePricer(LinearRegression().fit(frame[FEATURE_COLUMNS], frame.price))


@dataclass
class TextPricer:
    model: object
    vectorizer: object
    upper_bound: float | None = None

    @property
    def __name__(self):
        return f"text_{type(self.model).__name__}"

    def __call__(self, item):
        features = self.vectorizer.transform([item.summary or ""])
        value = max(0.0, float(self.model.predict(features)[0]))
        return min(value, self.upper_bound) if self.upper_bound is not None else value


def vectorize_training(train):
    vectorizer = CountVectorizer(max_features=2000, stop_words="english")
    features = vectorizer.fit_transform([item.summary or "" for item in train])
    return vectorizer, features, np.array([item.price for item in train], dtype=float)


def fit_text_model(vectorizer, features, prices, kind="linear", seed=42):
    if features.shape[0] != len(prices):
        raise ValueError("Features and prices must contain the same number of rows")
    if kind == "linear":
        model = LinearRegression()
    elif kind in ("forest", "full_forest"):
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1 if kind == "full_forest" else 4)
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=1000, random_state=seed, n_jobs=4, learning_rate=0.1)
    else:
        raise ValueError(f"Unknown model: {kind}")
    limit = 15_000 if kind == "forest" else len(prices)
    model.fit(features[:limit], prices[:limit])
    return TextPricer(model, vectorizer, 1000 if kind == "full_forest" else None)
