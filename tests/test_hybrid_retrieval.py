import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from agents.ensemble_agent import EnsembleAgent
from agents.deals import Deal, DealSelection
from agents.frontier_agent import FrontierAgent
from agents.guardrails import PriceGuardrailError, extract_single_price
from agents.planning_agent import PlanningAgent
from agents.rag_judge import RAGJudge, RAGJudgement
from agents.retrieval import (
    RetrievalCandidate,
    SQLiteBM25Index,
    reciprocal_rank_fusion,
    retrieval_metrics,
)


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def count(self):
        return len(self.records)

    def get(self, *, limit, offset, include):
        records = self.records[offset : offset + limit]
        return {
            "ids": [record[0] for record in records],
            "documents": [record[1] for record in records],
            "metadatas": [record[2] for record in records],
        }

    def query(self, *, query_embeddings, n_results):
        records = self.records[:n_results]
        return {
            "ids": [[record[0] for record in records]],
            "documents": [[record[1] for record in records]],
            "metadatas": [[record[2] for record in records]],
        }


class ExactMatchReranker:
    def score(self, query, documents):
        return [10.0 if "rtx 4090" in document.lower() else 0.0 for document in documents]


class HybridRetrievalTests(unittest.TestCase):
    @staticmethod
    def records():
        return [
            ("generic", "Powerful gaming desktop computer", {"price": 900}),
            ("exact", "Gaming desktop with NVIDIA RTX 4090 GPU", {"price": 2500}),
            ("other", "Office laptop with integrated graphics", {"price": 600}),
        ]

    def test_sqlite_bm25_builds_and_finds_exact_product_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SQLiteBM25Index(Path(directory) / "products.sqlite3")
            indexed = index.rebuild(FakeCollection(self.records()), batch_size=2)
            results = index.search("RTX 4090 gaming desktop", limit=2)

        self.assertEqual(indexed, 3)
        self.assertEqual(results[0].id, "exact")
        self.assertEqual(results[0].metadata["price"], 2500)

    def test_rank_fusion_rewards_candidates_found_by_both_retrievers(self):
        dense = [
            RetrievalCandidate("dense", "dense", dense_rank=1),
            RetrievalCandidate("both", "both", dense_rank=2),
        ]
        lexical = [
            RetrievalCandidate("both", "both", lexical_rank=1),
            RetrievalCandidate("lexical", "lexical", lexical_rank=2),
        ]

        fused = reciprocal_rank_fusion([dense, lexical])

        self.assertEqual(fused[0].id, "both")

    def test_frontier_combines_chroma_bm25_and_reranking(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = FakeCollection(self.records())
            index = SQLiteBM25Index(Path(directory) / "products.sqlite3")
            index.rebuild(collection)
            agent = FrontierAgent(
                collection,
                encoder=Mock(encode=Mock(return_value=np.array([[0.1, 0.2]]))),
                lexical_index=index,
                candidate_reranker=ExactMatchReranker(),
                candidate_count=3,
                result_count=2,
            )

            result = agent.retrieve("RTX 4090 gaming desktop")

        self.assertTrue(result.used_hybrid)
        self.assertTrue(result.reranked)
        self.assertEqual(result.candidates[0].id, "exact")

    def test_retrieval_metrics_are_objective_and_bounded(self):
        metrics = retrieval_metrics(["a", "b", "c"], {"b", "x"}, k=3)

        self.assertAlmostEqual(metrics["precision@3"], 1 / 3)
        self.assertEqual(metrics["recall@3"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertGreaterEqual(metrics["ndcg@3"], 0)
        self.assertLessEqual(metrics["ndcg@3"], 1)


class GuardrailTests(unittest.TestCase):
    def test_accepts_one_price_and_rejects_conflicting_prices(self):
        self.assertEqual(extract_single_price("$1,249.50"), 1249.50)
        with self.assertRaisesRegex(PriceGuardrailError, "multiple conflicting"):
            extract_single_price("Maybe $999 or $1,199")

    def test_ensemble_rejects_extreme_model_disagreement(self):
        agent = EnsembleAgent.__new__(EnsembleAgent)
        agent.preprocessor = Mock(preprocess=Mock(return_value="product"), model_name="test")
        agent.specialist = Mock(price=Mock(return_value=100))
        agent.frontier = Mock(price=Mock(return_value=110))
        agent.neural_network = Mock(price=Mock(return_value=10_000))
        agent.MAX_RELATIVE_SPREAD = 2.0

        with self.assertRaisesRegex(PriceGuardrailError, "disagree"):
            agent.price("product")

    def test_planner_rejects_large_absolute_but_small_percentage_discount(self):
        deal = Deal(
            product_description="Expensive product",
            price=950,
            url="https://example.com/product",
        )
        agent = PlanningAgent.__new__(PlanningAgent)
        agent.scanner = Mock(scan=Mock(return_value=DealSelection(deals=[deal])))
        agent.ensemble = Mock(price=Mock(return_value=1010))
        agent.messenger = Mock()

        result = agent.plan(memory=[])

        self.assertIsNone(result)
        agent.messenger.alert.assert_not_called()


class LLMJudgeTests(unittest.TestCase):
    def test_judge_returns_validated_structured_result(self):
        expected = RAGJudgement(
            relevance=3,
            same_product_category=True,
            specifications_compatible=True,
            condition_mismatch=False,
            rationale="Exact model and capacity match.",
        )
        response = Mock()
        response.choices = [
            Mock(message=Mock(content=expected.model_dump_json()))
        ]
        completion = Mock(return_value=response)
        judge = RAGJudge(completion_fn=completion)

        result = judge.judge("RTX 4090 desktop", "Desktop with RTX 4090")

        self.assertEqual(result, expected)
        self.assertEqual(completion.call_args.kwargs["response_format"], RAGJudgement)


if __name__ == "__main__":
    unittest.main()
