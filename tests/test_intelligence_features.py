import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from agents.deals import Deal, Opportunity
from agents.ensemble_agent import EnsembleAgent, PriceEstimate
from agents.retrieval import SQLiteBM25Index, expand_product_query
from agents.state_graph import DealStateGraph
from api import app
from pricer.intelligence_store import IntelligenceStore
from pricer.observability import Observability
from pricer.monitoring import drift_level, population_stability_index


def sample_opportunity() -> Opportunity:
    return Opportunity(
        deal=Deal(
            product_description="RTX 4090 gaming desktop",
            price=1200,
            url="https://example.com/deal?utm_source=test",
        ),
        estimate=2000,
        discount=800,
        model_estimates={"frontier_rag": 2100, "specialist": 1900},
        confidence=0.8,
        confidence_low=1700,
        confidence_high=2300,
        evidence=[{"id": "one", "price": 2050}],
    )


class FakeCollection:
    def __init__(self):
        self.records = [
            ("gpu", "RTX 4090 desktop", {"price": 2200, "category": "Electronics"}),
            ("chair", "Office chair", {"price": 200, "category": "Office"}),
        ]

    def count(self):
        return len(self.records)

    def get(self, *, limit, offset, include):
        rows = self.records[offset : offset + limit]
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }


class IntelligenceStoreTests(unittest.TestCase):
    def test_feedback_history_searches_checkpoints_and_metrics_are_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "intelligence.sqlite3")
            opportunity = sample_opportunity()
            store.upsert_opportunity(opportunity)
            updated = store.record_feedback(
                opportunity.deal.url,
                "approved",
                corrected_price=2100,
                reason="Comparable evidence checks out",
            )
            search_id = store.save_search("GPU watch", "RTX 4090", {})
            store.metric("retrieval.ndcg", 0.9, version="hybrid-v2")
            store.save_checkpoint("run-1", {"status": "approval_required"})

            self.assertEqual(updated.status, "approved")
            self.assertEqual(store.list_opportunities("approved")[0].status, "approved")
            self.assertEqual(store.price_history(opportunity.deal.url)[0]["observed_price"], 1200)
            self.assertEqual(store.list_saved_searches()[0]["query"], "RTX 4090")
            self.assertEqual(store.metric_summary()[0]["average"], 0.9)
            self.assertEqual(store.load_checkpoint("run-1")["status"], "approval_required")
            self.assertEqual(store.price_insights(opportunity.deal.url)["recommendation"], "buy_now")
            evaluation = store.evaluation_summary()
            self.assertEqual(evaluation["feedback_count"], 1)
            self.assertEqual(evaluation["approved_count"], 1)
            self.assertEqual(evaluation["rejected_count"], 0)
            self.assertEqual(evaluation["mean_absolute_error"], 100)
            self.assertEqual(evaluation["root_mean_squared_error"], 100)
            self.assertEqual(evaluation["mean_signed_error"], -100)
            self.assertEqual(store.saved_search_matches(search_id)[0].deal.price, 1200)

            # A later rescan must not change the estimate used for old feedback.
            opportunity.estimate = 5000
            store.upsert_opportunity(opportunity)
            self.assertEqual(store.evaluation_summary()["mean_absolute_error"], 100)

    def test_feedback_rejects_invalid_corrected_price(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "intelligence.sqlite3")
            opportunity = sample_opportunity()
            store.upsert_opportunity(opportunity)

            with self.assertRaises(ValueError):
                store.record_feedback(
                    opportunity.deal.url,
                    "rejected",
                    corrected_price=-1,
                    reason="invalid label",
                )

    def test_observability_records_success_error_and_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "metrics.sqlite3")
            telemetry = Observability(store)
            with telemetry.trace("work", component="test"):
                pass
            with self.assertRaises(RuntimeError):
                with telemetry.trace("broken"):
                    raise RuntimeError("expected")
            names = {row["name"] for row in store.metric_summary()}
            self.assertTrue({"work.success", "work.latency_ms", "broken.errors"} <= names)


class EstimateAndGraphTests(unittest.TestCase):
    def test_detailed_ensemble_keeps_scalar_interface(self):
        agent = EnsembleAgent.__new__(EnsembleAgent)
        agent.preprocessor = Mock(preprocess=Mock(return_value="gpu"), model_name="test")
        agent.specialist = Mock(price=Mock(return_value=1900))
        agent.frontier = Mock(
            price=Mock(return_value=2100),
            evidence=Mock(return_value=[{"id": "gpu", "price": 2050}]),
        )
        agent.neural_network = Mock(price=Mock(return_value=2000))
        agent.MAX_RELATIVE_SPREAD = 2.0

        result = agent.estimate("RTX 4090")

        self.assertAlmostEqual(result.value, 2070)
        self.assertGreater(result.confidence, 0)
        self.assertLess(result.confidence_low, result.value)
        self.assertGreater(result.confidence_high, result.value)
        self.assertEqual(result.evidence[0]["id"], "gpu")

    def test_graph_checkpoints_and_resumes_human_approval(self):
        estimate = PriceEstimate(
            value=2000,
            model_estimates={"test": 2000},
            confidence=0.7,
            confidence_low=1700,
            confidence_high=2300,
            evidence=[],
            explanation="test estimate",
        )
        ensemble = Mock(estimate=Mock(return_value=estimate))
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "graph.sqlite3")
            graph = DealStateGraph(ensemble, store)
            state = graph.run(sample_opportunity().deal, run_id="graph-run")
            resumed = graph.resume_approval("graph-run", "approved", reason="verified")

            self.assertEqual(state["status"], "approval_required")
            self.assertEqual(resumed["status"], "approved")
            self.assertEqual(store.load_checkpoint("graph-run")["status"], "approved")


class AdvancedRetrievalTests(unittest.TestCase):
    def test_query_expansion_preserves_model_tokens(self):
        variants = expand_product_query("A gaming desktop with RTX 4090")
        self.assertIn("rtx 4090", variants)
        self.assertGreaterEqual(len(variants), 2)

    def test_bm25_metadata_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteBM25Index(Path(directory) / "products.sqlite3")
            index.rebuild(FakeCollection())
            result = index.search(
                "RTX 4090 desktop", metadata_filter={"category": "Electronics"}
            )
            excluded = index.search(
                "RTX 4090 desktop", metadata_filter={"category": "Office"}
            )
            self.assertEqual(result[0].id, "gpu")
            self.assertEqual(excluded, [])

    def test_distribution_drift_monitoring(self):
        stable = population_stability_index([1, 2, 3, 4], [1, 2, 3, 4])
        shifted = population_stability_index([1, 1, 2, 2], [8, 9, 9, 10])
        self.assertEqual(drift_level(stable), "stable")
        self.assertEqual(drift_level(shifted), "drifted")


class ApiTests(unittest.TestCase):
    def test_health_is_public_and_configured_api_key_is_enforced(self):
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        with patch.dict(os.environ, {"PRICER_API_KEY": "secret"}):
            response = client.get("/opportunities")
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
