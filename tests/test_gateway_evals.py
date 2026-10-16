"""Offline checks for the provider and gateway evaluation runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pricer.gateway_evals.runner import (
    Target,
    _check_output,
    load_cases,
    load_targets,
    run_benchmark,
)


class GatewayEvalConfigTests(unittest.TestCase):
    def test_case_loader_validates_price_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "price",
                        "messages": [{"role": "user", "content": "prompt"}],
                        "checks": {"price": {"actual_price": float("nan")}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                load_cases(path)

    def test_target_url_cannot_embed_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.toml"
            path.write_text(
                '[targets.bad]\nmodel = "test"\nbase_url = "https://user:secret@example.com/v1"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "credentials"):
                load_targets(path)


class GatewayEvalMetricTests(unittest.TestCase):
    def test_output_checks_detect_prices_and_malformed_answers(self):
        checks = {"nonempty": True, "regex": "price", "price": {"minimum": 1, "maximum": 500}}
        self.assertEqual(_check_output("Price: $125", checks), (None, 125.0))
        self.assertEqual(_check_output("No estimate", checks)[0], "regex_mismatch")
        self.assertEqual(_check_output("Price: unclear", checks)[0], "invalid_price")
        self.assertEqual(
            _check_output("Price: $125 or $150", checks)[0], "invalid_price"
        )

    def test_report_computes_paired_mae_and_omits_prompt_text(self):
        targets = {
            name: Target(name, "test", "openai-chat-completions", "http://localhost/v1", name, None)
            for name in ("base", "candidate")
        }
        private_prompt = "customer product details that must not be saved"
        cases = [
            {
                "id": "price-1",
                "messages": [{"role": "user", "content": private_prompt}],
                "checks": {"price": {"actual_price": 100}},
            }
        ]

        def response(target, case, timeout, request_id):
            price = 110 if target.name == "base" else 105
            return {
                "case_id": case["id"],
                "request_id": request_id,
                "status": "ok",
                "latency_ms": 20.0,
                "error_code": None,
                "http_status": 200,
                "response_model": target.model,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "estimated_cost_usd": None,
                "price": price,
                "response_headers": {},
            }

        with patch("pricer.gateway_evals.runner._request", side_effect=response), patch(
            "pricer.gateway_evals.runner._gpu_name", return_value=None
        ):
            report = run_benchmark(
                targets,
                cases,
                request_count=2,
                concurrency=1,
                warmups=0,
                timeout=5,
                max_failure_rate=0,
                max_invalid_rate=0,
                max_paired_mae_regression=0,
            )

        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["targets"][0]["pricing_quality"]["mae"], 10)
        self.assertEqual(report["targets"][1]["pricing_quality"]["mae"], 5)
        paired = report["paired_pricing_quality"][0]
        self.assertEqual(paired["mae_delta_candidate_minus_reference"], -5)
        self.assertNotIn(private_prompt, json.dumps(report))


if __name__ == "__main__":
    unittest.main()
