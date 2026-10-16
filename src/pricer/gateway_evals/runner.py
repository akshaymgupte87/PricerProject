"""Run reproducible, content-redacted checks against chat-completion targets."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


PRICE_PATTERN = re.compile(r"(?<![\w.])\$?\s*(\d[\d,]*(?:\.\d+)?)")
SAFE_RESPONSE_HEADERS = (
    "x-litellm-model-id",
    "x-litellm-response-cost",
    "x-litellm-call-id",
    "x-request-id",
)


@dataclass(frozen=True)
class Target:
    name: str
    provider: str
    protocol: str
    base_url: str
    model: str
    api_key: str | None
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None


def _safe_endpoint(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("target base_url must be an absolute HTTP(S) URL")
    if parts.username or parts.password:
        raise ValueError("put credentials in environment variables, not the target URL")
    netloc = parts.hostname
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path.rstrip("/"), "", ""))


def load_targets(path: Path) -> dict[str, Target]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    targets: dict[str, Target] = {}
    for name, raw in config.get("targets", {}).items():
        if not isinstance(raw, dict):
            raise ValueError(f"target {name!r} must be a TOML table")
        base_url = os.getenv(raw.get("base_url_env", ""), "").strip()
        base_url = base_url or str(raw.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"target {name!r} must set base_url or base_url_env")
        api_key = None
        for env_name in raw.get("api_key_envs", []):
            if os.getenv(env_name, "").strip():
                api_key = os.environ[env_name].strip()
                break
        costs = raw.get("cost_per_million_tokens", {})
        protocol = str(raw.get("protocol", "openai-chat-completions"))
        if protocol not in {"openai-chat-completions", "ollama-native"}:
            raise ValueError(f"target {name!r} uses unsupported protocol {protocol!r}")
        targets[name] = Target(
            name=name,
            provider=str(raw.get("provider", name)),
            protocol=protocol,
            base_url=_safe_endpoint(base_url),
            model=str(raw["model"]),
            api_key=api_key,
            input_cost_per_million=_optional_float(costs.get("input")),
            output_cost_per_million=_optional_float(costs.get("output")),
        )
    if not targets:
        raise ValueError(f"no [targets.*] entries found in {path}")
    return targets


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("token prices must be finite, non-negative numbers")
    return number


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError(f"no evaluation cases found in {path}")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each evaluation case must be a JSON object")
        case_id = str(case.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError("case IDs must be present and unique")
        seen.add(case_id)
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {case_id!r} needs a non-empty messages list")
        if not all(
            isinstance(row, dict)
            and isinstance(row.get("role"), str)
            and row["role"].strip()
            and "content" in row
            for row in messages
        ):
            raise ValueError(f"case {case_id!r} has an invalid message")
        if not isinstance(case.get("request", {}), dict):
            raise ValueError(f"case {case_id!r} request must be an object")
        if not isinstance(case.get("checks", {}), dict):
            raise ValueError(f"case {case_id!r} checks must be an object")
        price_check = case.get("checks", {}).get("price")
        if price_check is not None:
            if not isinstance(price_check, dict):
                raise ValueError(f"case {case_id!r} price check must be an object")
            if price_check.get("actual_price") is not None:
                actual = float(price_check["actual_price"])
                if not math.isfinite(actual) or actual <= 0:
                    raise ValueError(f"case {case_id!r} actual_price must be finite and positive")
            minimum = _optional_float(price_check.get("minimum", 0.01))
            maximum = _optional_float(price_check.get("maximum", 1_000_000))
            if minimum is None or maximum is None or minimum > maximum:
                raise ValueError(f"case {case_id!r} price bounds must be finite and ordered")
    return cases


def _request(
    target: Target,
    case: dict[str, Any],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    if target.protocol == "ollama-native":
        request_options = dict(case.get("request", {}))
        options = {}
        for key in ("temperature", "top_p", "top_k", "seed", "num_predict", "repeat_penalty"):
            if key in request_options:
                options[key] = request_options.pop(key)
        if "max_tokens" in request_options:
            options["num_predict"] = request_options.pop("max_tokens")
        payload = {
            "model": target.model,
            "messages": case["messages"],
            "stream": False,
            "options": options,
            **request_options,
        }
        request_url = f"{target.base_url}/api/chat"
    elif target.protocol == "openai-chat-completions":
        payload = {"model": target.model, "messages": case["messages"], **case.get("request", {})}
        payload["model"] = target.model
        payload["messages"] = case["messages"]
        request_url = f"{target.base_url}/chat/completions"
    else:
        raise ValueError(f"unsupported target protocol {target.protocol!r}")
    headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    request = urllib.request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    response_headers: dict[str, str] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            response_headers = {
                name: response.headers.get(name, "")
                for name in SAFE_RESPONSE_HEADERS
                if response.headers.get(name)
            }
            status_code = response.status
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _failure(case, request_id, elapsed_ms, "invalid_json", status_code, response_headers)
        if target.protocol == "ollama-native":
            content = (decoded.get("message") or {}).get("content")
            response_model = decoded.get("model")
            prompt_tokens = _safe_int(decoded.get("prompt_eval_count"))
            completion_tokens = _safe_int(decoded.get("eval_count"))
            total_tokens = (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            )
        else:
            choices = decoded.get("choices") or []
            content = None
            if choices and isinstance(choices[0], dict):
                content = (choices[0].get("message") or {}).get("content")
            response_model = decoded.get("model")
            usage = decoded.get("usage") or {}
            prompt_tokens = _safe_int(usage.get("prompt_tokens"))
            completion_tokens = _safe_int(usage.get("completion_tokens"))
            total_tokens = _safe_int(usage.get("total_tokens"))
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
        if not isinstance(content, str):
            return _failure(case, request_id, elapsed_ms, "invalid_response", status_code, response_headers)
        if not content.strip():
            return _failure(case, request_id, elapsed_ms, "empty_response", status_code, response_headers)
        check_error, parsed_price = _check_output(content, case.get("checks", {}))
        cost = _response_cost(response_headers)
        if cost is None and prompt_tokens is not None and completion_tokens is not None:
            if target.input_cost_per_million is not None and target.output_cost_per_million is not None:
                cost = (
                    prompt_tokens * target.input_cost_per_million
                    + completion_tokens * target.output_cost_per_million
                ) / 1_000_000
        result = {
            "case_id": case["id"],
            "request_id": request_id,
            "status": "check_failed" if check_error else "ok",
            "latency_ms": round(elapsed_ms, 3),
            "error_code": check_error,
            "http_status": status_code,
            "response_model": response_model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "estimated_cost_usd": cost,
            "price": parsed_price,
            "response_headers": response_headers,
        }
        return result
    except urllib.error.HTTPError as error:
        response_headers = {
            name: error.headers.get(name, "")
            for name in SAFE_RESPONSE_HEADERS
            if error.headers and error.headers.get(name)
        }
        return _failure(
            case,
            request_id,
            (time.perf_counter() - started) * 1000,
            f"http_{error.code}",
            error.code,
            response_headers,
        )
    except (TimeoutError, socket.timeout):
        return _failure(case, request_id, (time.perf_counter() - started) * 1000, "timeout")
    except urllib.error.URLError as error:
        reason = error.reason
        code = "timeout" if isinstance(reason, (TimeoutError, socket.timeout)) else "connection_error"
        return _failure(case, request_id, (time.perf_counter() - started) * 1000, code)
    except (OSError, ValueError):
        return _failure(case, request_id, (time.perf_counter() - started) * 1000, "request_error")


def _failure(
    case: dict[str, Any],
    request_id: str,
    elapsed_ms: float,
    code: str,
    status: int | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "request_id": request_id,
        "status": "request_failed" if code.startswith(("http_", "timeout", "connection", "request")) else "invalid_output",
        "latency_ms": round(elapsed_ms, 3),
        "error_code": code,
        "http_status": status,
        "response_model": None,
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "estimated_cost_usd": None,
        "price": None,
        "response_headers": headers or {},
    }


def _check_output(content: str, checks: dict[str, Any]) -> tuple[str | None, float | None]:
    if checks.get("nonempty", True) and not content.strip():
        return "empty_response", None
    patterns = checks.get("regex")
    if patterns:
        patterns = [patterns] if isinstance(patterns, str) else patterns
        if not all(re.search(str(pattern), content, flags=re.IGNORECASE) for pattern in patterns):
            return "regex_mismatch", None
    if checks.get("json"):
        try:
            json.loads(content)
        except json.JSONDecodeError:
            return "invalid_json_content", None
    price_check = checks.get("price")
    if price_check is None:
        return None, None
    values = [float(value.replace(",", "")) for value in PRICE_PATTERN.findall(content)]
    if len(set(values)) != 1 or not values:
        return "invalid_price", None
    price = values[0]
    minimum = float(price_check.get("minimum", 0.01))
    maximum = float(price_check.get("maximum", 1_000_000))
    if not math.isfinite(price) or not minimum <= price <= maximum:
        return "price_out_of_range", price
    return None, price


def _safe_int(value: Any) -> int | None:
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def _response_cost(headers: dict[str, str]) -> float | None:
    for name, value in headers.items():
        if name.lower() == "x-litellm-response-cost":
            try:
                number = float(value)
                return number if math.isfinite(number) and number >= 0 else None
            except ValueError:
                return None
    return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = max(1, math.ceil(percentile * len(values)))
    return values[rank - 1]


def _gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if names else None


def _target_report(
    target: Target,
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    wall_seconds: float,
    warmup_errors: list[str],
) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    successes = [row for row in rows if row["status"] == "ok"]
    token_values = [row["usage"]["total_tokens"] for row in rows if row["usage"]["total_tokens"] is not None]
    costs = [row["estimated_cost_usd"] for row in rows]
    prices: list[tuple[float, float]] = []
    case_lookup = {case["id"]: case for case in cases}
    for row in successes:
        price_check = case_lookup[row["case_id"]].get("checks", {}).get("price") or {}
        expected = price_check.get("actual_price")
        if expected is not None and row["price"] is not None:
            prices.append((float(expected), float(row["price"])))
    price_errors = [abs(actual - predicted) for actual, predicted in prices]
    errors = Counter(row["error_code"] for row in rows if row["error_code"])
    return {
        "target": target.name,
        "provider": target.provider,
        "protocol": target.protocol,
        "endpoint": target.base_url,
        "model_alias": target.model,
        "attempted": len(rows),
        "successful": len(successes),
        "failed_requests": sum(row["status"] == "request_failed" for row in rows),
        "invalid_outputs": sum(row["status"] in {"invalid_output", "check_failed"} for row in rows),
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "error_counts": dict(errors),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(len(rows) / wall_seconds, 3) if wall_seconds else None,
        "total_tokens": sum(token_values) if token_values else None,
        "token_count_available_requests": len(token_values),
        "estimated_cost_usd": round(sum(costs), 10) if costs and all(cost is not None for cost in costs) else None,
        "cost_available_requests": sum(cost is not None for cost in costs),
        "pricing_quality": {
            "n": len(prices),
            "mae": round(sum(price_errors) / len(prices), 4) if prices else None,
            "wape": round(sum(price_errors) / sum(actual for actual, _ in prices), 6) if prices else None,
        },
        "warmup_errors": warmup_errors,
        "requests": rows,
    }


def run_benchmark(
    targets: dict[str, Target],
    cases: list[dict[str, Any]],
    *,
    request_count: int,
    concurrency: int,
    warmups: int,
    timeout: float,
    max_failure_rate: float,
    max_invalid_rate: float,
    max_mae: float | None = None,
    max_wape: float | None = None,
    max_paired_mae_regression: float | None = None,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if request_count < 1 or concurrency < 1 or warmups < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("requests/concurrency/timeout must be positive and warmups non-negative")
    if not 0 <= max_failure_rate <= 1 or not 0 <= max_invalid_rate <= 1:
        raise ValueError("failure-rate thresholds must be between 0 and 1")
    if (
        (max_mae is not None and (not math.isfinite(max_mae) or max_mae < 0))
        or (max_wape is not None and (not math.isfinite(max_wape) or max_wape < 0))
        or (
            max_paired_mae_regression is not None
            and (not math.isfinite(max_paired_mae_regression) or max_paired_mae_regression < 0)
        )
    ):
        raise ValueError("pricing error thresholds must be non-negative")
    if max_mae is not None or max_wape is not None or max_paired_mae_regression is not None:
        if not any(
            (case.get("checks", {}).get("price") or {}).get("actual_price") is not None
            for case in cases
        ):
            raise ValueError("MAE/WAPE gates require cases with checks.price.actual_price labels")
        if max_paired_mae_regression is not None and len(targets) < 2:
            raise ValueError("paired MAE regression gate requires at least two targets")
    target_results = []
    gate_failures: list[str] = []
    for target in targets.values():
        warmup_errors: list[str] = []
        for index in range(warmups):
            row = _request(target, cases[index % len(cases)], timeout, f"warmup-{uuid.uuid4().hex}")
            if row["status"] != "ok":
                warmup_errors.append(str(row["error_code"]))
        scheduled = [
            (cases[index % len(cases)], f"gateway-eval-{uuid.uuid4().hex}")
            for index in range(request_count)
        ]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_request, target, case, timeout, request_id)
                for case, request_id in scheduled
            ]
            rows = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        summary = _target_report(target, cases, rows, wall_seconds, warmup_errors)
        if summary["failed_requests"] / request_count > max_failure_rate:
            gate_failures.append(f"{target.name}: request failure rate exceeded threshold")
        if summary["invalid_outputs"] / request_count > max_invalid_rate:
            gate_failures.append(f"{target.name}: invalid-output rate exceeded threshold")
        quality = summary["pricing_quality"]
        if max_mae is not None and (quality["mae"] is None or quality["mae"] > max_mae):
            gate_failures.append(f"{target.name}: pricing MAE exceeded threshold or had no valid labels")
        if max_wape is not None and (quality["wape"] is None or quality["wape"] > max_wape):
            gate_failures.append(f"{target.name}: pricing WAPE exceeded threshold or had no valid labels")
        target_results.append(summary)
    paired_quality: list[dict[str, Any]] = []
    if target_results:
        reference = target_results[0]
        reference_rows = reference["requests"]
        for target_result in target_results[1:]:
            deltas: list[float] = []
            for reference_row, candidate_row in zip(reference_rows, target_result["requests"]):
                case = next(item for item in cases if item["id"] == reference_row["case_id"])
                actual = (case.get("checks", {}).get("price") or {}).get("actual_price")
                if (
                    actual is not None
                    and reference_row["price"] is not None
                    and candidate_row["price"] is not None
                ):
                    deltas.append(
                        abs(float(actual) - float(candidate_row["price"]))
                        - abs(float(actual) - float(reference_row["price"]))
                    )
            if deltas:
                rng = random.Random(42)
                samples = sorted(
                    sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                    for _ in range(2000)
                )
                mean_delta = sum(deltas) / len(deltas)
                interval = [samples[49], samples[1949]]
            else:
                mean_delta = None
                interval = None
            paired_quality.append(
                {
                    "reference_target": reference["target"],
                    "candidate_target": target_result["target"],
                    "n": len(deltas),
                    "mae_delta_candidate_minus_reference": mean_delta,
                    "bootstrap_95pct_ci": interval,
                    "interpretation": "Negative values favor the candidate target.",
                }
            )
            if max_paired_mae_regression is not None:
                if mean_delta is None or mean_delta > max_paired_mae_regression:
                    gate_failures.append(
                        f"{target_result['target']}: paired MAE regression exceeded threshold or had no paired cases"
                    )
    return {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "gpu": _gpu_name(),
        },
        "run": {
            "requests_per_target": request_count,
            "concurrency": concurrency,
            "warmups_per_target": warmups,
            "timeout_seconds": timeout,
            "case_ids": [case["id"] for case in cases],
            "response_text_recorded": False,
            "prompt_text_recorded": False,
        },
        "targets": target_results,
        "paired_pricing_quality": paired_quality,
        "thresholds": {
            "max_failure_rate": max_failure_rate,
            "max_invalid_rate": max_invalid_rate,
            "max_mae": max_mae,
            "max_wape": max_wape,
            "max_paired_mae_regression": max_paired_mae_regression,
        },
        "gate": {"passed": not gate_failures, "failures": gate_failures},
    }
