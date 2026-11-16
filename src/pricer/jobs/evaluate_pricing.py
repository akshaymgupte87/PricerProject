"""Train legacy and improved profiles and compare them on one held-out split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from pricer.agents.deep_neural_network import DeepNeuralNetwork

from pricer.data.items import Item
from pricer.jobs import train_neural


def _valid(items):
    return [item for item in items if item.summary and math.isfinite(item.price) and item.price >= 0]


def _predict(checkpoint, items):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    vectorizer = HashingVectorizer(**saved["vectorizer_config"])
    model = DeepNeuralNetwork(**saved["model_config"])
    model.load_state_dict(saved["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    target_mean, target_std = saved["target_mean"], saved["target_std"]
    model.eval()
    result = []
    latencies = []
    with torch.inference_mode():
        for item in items:
            started = time.perf_counter()
            features = vectorizer.transform([item.summary]).toarray().astype(np.float32, copy=False)
            tensor = torch.from_numpy(features).to(device)
            prediction = model(tensor).reshape(-1)[0].item()
            price = max(0.0, math.expm1(prediction * target_std + target_mean))
            latencies.append((time.perf_counter() - started) * 1000)
            result.append(price)
    return result, latencies


def _metrics(items, predictions, latencies):
    actual = np.asarray([item.price for item in items], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    errors = np.abs(actual - predicted)
    return {
        "n": len(items),
        "mae": float(errors.mean()),
        "rmse": float(np.sqrt(np.square(actual - predicted).mean())),
        "wape": float(errors.sum() / actual.sum()) if actual.sum() else None,
        "median_absolute_error": float(np.median(errors)),
        "failed_predictions": int((~np.isfinite(predicted)).sum()),
        "failure_rate": float((~np.isfinite(predicted)).mean()),
        "latency_ms_per_example_mean": float(np.mean(latencies)),
        "latency_ms_per_example_p95": float(np.percentile(latencies, 95)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ed-donner/items_full")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/pricing-holdout.json"))
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--max-test", type=int)
    args = parser.parse_args(argv)
    from datasets import load_dataset
    from pricer.data.items import Item
    dataset = load_dataset(args.dataset)
    train = [Item.model_validate(row) for row in dataset["train"]]
    validation = [Item.model_validate(row) for row in dataset["validation"]]
    test = [Item.model_validate(row) for row in dataset["test"]]
    train = _valid(train)[:args.max_train]
    validation = _valid(validation)[:args.max_validation]
    test = _valid(test)[:args.max_test]
    if not train or not validation or not test:
        raise ValueError("training, validation, and held-out test splits must all contain valid examples")

    profiles = {}
    with tempfile.TemporaryDirectory(prefix="pricer-eval-") as temporary:
        for name in ("legacy", "improved"):
            checkpoint = Path(temporary) / f"{name}.pth"
            train_neural.train_profile(args.dataset, name, checkpoint, train, validation)
            predictions, latencies = _predict(checkpoint, test)
            profiles[name] = _metrics(test, predictions, latencies)

    report = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split_sizes_used": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_split": "test",
        "profiles": profiles,
        "mae_delta_improved_minus_legacy": profiles["improved"]["mae"] - profiles["legacy"]["mae"],
        "note": "No test examples are used for fitting or checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
