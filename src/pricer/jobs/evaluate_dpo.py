"""Compare a DPO adapter with its base model on independent labelled cases."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from pricer.agents.deal_seen_store import canonicalize_url
from pricer.inference.finetuned import build_prompt, extract_price


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    required = {"id", "url", "product_description", "actual_price"}
    seen: set[str] = set()
    for row in cases:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Evaluation case is missing fields: {sorted(missing)}")
        case_id = str(row["id"])
        if not case_id or case_id in seen:
            raise ValueError(f"Evaluation case IDs must be non-empty and unique: {case_id!r}")
        seen.add(case_id)
        price = float(row["actual_price"])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"actual_price must be finite and positive for case {case_id}")
    return cases


def reject_training_overlap(cases: list[dict[str, Any]], training_data: Path) -> None:
    if not training_data.is_file():
        raise FileNotFoundError(f"Training preference data not found: {training_data}")
    trained_ids: set[str] = set()
    trained_urls: set[str] = set()
    for line in training_data.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("url"):
                trained_urls.add(canonicalize_url(str(row["url"])))
            if row.get("id"):
                trained_ids.add(str(row["id"]))
    overlap = [str(row["id"]) for row in cases if str(row["id"]) in trained_ids]
    overlap += [
        str(row["id"])
        for row in cases
        if canonicalize_url(str(row["url"])) in trained_urls
    ]
    if overlap:
        raise ValueError(
            "Evaluation cases overlap with DPO training data; use product IDs/URLs "
            f"that were excluded from training: {sorted(set(overlap))[:10]}"
        )


def _load_model(base_model: str, adapter: Path | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("DPO comparison requires a CUDA-enabled PyTorch environment")
    use_bf16 = torch.cuda.is_bf16_supported()
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quantization, device_map="auto"
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def _predict(model, tokenizer, description: str) -> float:
    import torch

    prompt = build_prompt(description)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_price(text)


def _metrics(
    actual: list[float], predicted: list[float], elapsed: list[float], total_cases: int
) -> dict[str, float | int]:
    errors = [abs(y - p) for y, p in zip(actual, predicted)]
    return {
        "n_valid": len(actual),
        "n_total": total_cases,
        "valid_output_rate": len(predicted) / total_cases if total_cases else 0.0,
        "mae": statistics.mean(errors),
        "rmse": math.sqrt(statistics.mean((y - p) ** 2 for y, p in zip(actual, predicted))),
        "wape": sum(errors) / sum(actual),
        "median_absolute_percentage_error": statistics.median(
            error / y for error, y in zip(errors, actual)
        ),
        "mean_latency_seconds": statistics.mean(elapsed),
        "p95_latency_seconds": sorted(elapsed)[
            min(len(elapsed) - 1, math.ceil(len(elapsed) * 0.95) - 1)
        ],
    }


def _evaluate_model(cases: list[dict[str, Any]], base_model: str, adapter: Path | None):
    model, tokenizer = _load_model(base_model, adapter)
    actual: list[float] = []
    predicted: list[float] = []
    elapsed: list[float] = []
    rows = []
    for case in cases:
        started = time.perf_counter()
        try:
            value = _predict(model, tokenizer, str(case["product_description"]))
            valid = True
        except (ValueError, RuntimeError, TypeError):
            value = None
            valid = False
        elapsed.append(time.perf_counter() - started)
        truth = float(case["actual_price"])
        if valid:
            actual.append(truth)
            predicted.append(float(value))
        rows.append({"id": str(case["id"]), "actual_price": truth, "predicted_price": value})
    del model, tokenizer
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    if predicted:
        metrics = _metrics(actual, predicted, elapsed, len(cases))
    else:
        metrics = {
            "n_valid": 0,
            "n_total": len(cases),
            "valid_output_rate": 0.0,
            "mean_latency_seconds": statistics.mean(elapsed),
            "p95_latency_seconds": sorted(elapsed)[
                min(len(elapsed) - 1, math.ceil(len(elapsed) * 0.95) - 1)
            ],
        }
    return metrics, rows


def evaluate(cases_path: Path, training_data: Path, base_model: str, adapter: Path, output: Path) -> dict:
    cases = load_cases(cases_path)
    reject_training_overlap(cases, training_data)
    base_metrics, base_rows = _evaluate_model(cases, base_model, None)
    adapter_metrics, adapter_rows = _evaluate_model(cases, base_model, adapter)
    paired = [
        (
            float(case["actual_price"]),
            float(base["predicted_price"]),
            float(tuned["predicted_price"]),
        )
        for case, base, tuned in zip(cases, base_rows, adapter_rows)
        if base["predicted_price"] is not None and tuned["predicted_price"] is not None
    ]
    differences = [abs(y - tuned) - abs(y - base) for y, base, tuned in paired]
    if differences:
        rng = random.Random(42)
        bootstrapped = sorted(
            statistics.mean(rng.choices(differences, k=len(differences)))
            for _ in range(2000)
        )
        confidence_interval = [bootstrapped[49], bootstrapped[1949]]
        mean_change = statistics.mean(differences)
    else:
        confidence_interval = None
        mean_change = None
    report = {
        "dataset": str(cases_path),
        "training_data": str(training_data),
        "base_model": base_model,
        "adapter": str(adapter),
        "n": len(cases),
        "base": base_metrics,
        "adapter_metrics": adapter_metrics,
        "paired_n": len(paired),
        "paired_mae_change_adapter_minus_base": mean_change,
        "paired_mae_change_bootstrap_95pct_ci": confidence_interval,
        "interpretation": (
            "Negative paired MAE change favors the adapter; this is a held-out "
            "experiment, not proof of generalization."
        ),
        "predictions": [
            {
                "id": case["id"],
                "actual_price": case["actual_price"],
                "base_price": base["predicted_price"],
                "adapter_price": tuned["predicted_price"],
            }
            for case, base, tuned in zip(cases, base_rows, adapter_rows)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
