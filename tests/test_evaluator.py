import unittest
from types import SimpleNamespace

from pricer.evaluation import evaluator as canonical_evaluator


class EvaluatorConsolidationTests(unittest.TestCase):
    def test_canonical_evaluator_accepts_domain_and_legacy_rows(self):
        domain = SimpleNamespace(title="Domain product", price=125)
        legacy = {
            "prompt": "Title: Legacy product\nDescription: example",
            "completion": "250",
        }
        tester = canonical_evaluator.Tester(lambda row: 100, [domain, legacy], size=2)

        self.assertEqual(tester.run_datapoint(0)[:3], ("Domain product", 100, 125.0))
        self.assertEqual(tester.run_datapoint(1)[:3], ("Legacy product", 100, 250.0))

    def test_sequential_evaluation_is_explicit(self):
        tester = canonical_evaluator.Tester(lambda row: 100, [], size=0, workers=1)
        self.assertIsInstance(tester, canonical_evaluator.Tester)
        self.assertEqual(tester.workers, 1)

    def test_retrieval_metrics_are_aggregated_canonically(self):
        metrics = canonical_evaluator.aggregate_retrieval_metrics(
            [(["a", "b"], {"a"}), (["x", "b"], {"b"})], k=2
        )
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["precision@2"], 0.5)


if __name__ == "__main__":
    unittest.main()
