"""Command-line entry point for provider and gateway evaluation runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pricer.config import Settings, load_environment
from pricer.gateway_evals.runner import load_cases, load_targets, run_benchmark


def main() -> None:
    load_environment()
    settings = Settings.from_env()
    default_root = settings.home / "benchmarks" / "gateway"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_root / "targets.toml")
    parser.add_argument("--cases", type=Path, default=default_root / "cases.jsonl")
    parser.add_argument("--targets", help="Comma-separated names from the TOML config; defaults to all.")
    parser.add_argument("--requests", type=int, default=50, help="Measured requests per target.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-failure-rate", type=float, default=0)
    parser.add_argument("--max-invalid-rate", type=float, default=0)
    parser.add_argument("--max-mae", type=float)
    parser.add_argument("--max-wape", type=float)
    parser.add_argument("--max-paired-mae-regression", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    available = load_targets(args.config)
    if args.targets:
        names = [name.strip() for name in args.targets.split(",") if name.strip()]
        missing = sorted(set(names) - available.keys())
        if missing:
            parser.error(f"unknown target(s) {missing}; available: {sorted(available)}")
        targets = {name: available[name] for name in names}
    else:
        targets = available
    cases = load_cases(args.cases)
    report = run_benchmark(
        targets,
        cases,
        request_count=args.requests,
        concurrency=args.concurrency,
        warmups=args.warmups,
        timeout=args.timeout,
        max_failure_rate=args.max_failure_rate,
        max_invalid_rate=args.max_invalid_rate,
        max_mae=args.max_mae,
        max_wape=args.max_wape,
        max_paired_mae_regression=args.max_paired_mae_regression,
    )
    output = args.output or (
        settings.home
        / "artifacts"
        / "gateway-evals"
        / f"gateway-eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = [
        {
            "target": row["target"],
            "success_rate": row["success_rate"],
            "failed_requests": row["failed_requests"],
            "invalid_outputs": row["invalid_outputs"],
            "p50_ms": row["latency_ms"]["p50"],
            "p95_ms": row["latency_ms"]["p95"],
            "requests_per_second": row["requests_per_second"],
            "estimated_cost_usd": row["estimated_cost_usd"],
        }
        for row in report["targets"]
    ]
    print(json.dumps({"gate": report["gate"], "summary": summary, "report": str(output)}, indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
