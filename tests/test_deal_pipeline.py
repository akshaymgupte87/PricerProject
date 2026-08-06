import json
import queue
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import chromadb

from agents.deal_seen_store import DealSeenStore, canonicalize_url
from agents.deals import Deal, DealSelection, Opportunity
from agents.planning_agent import PlanningAgent
from agents.scanner_agent import ScannerAgent
from agents.specialist_agent import SpecialistAgent
from deal_agent_framework import DealAgentFramework
from pricer_ephemeral import build_prompt, price as fine_tuned_price
from price_is_right import run_framework_worker, table_for


def opportunity(name: str, price: float, estimate: float) -> Opportunity:
    return Opportunity(
        deal=Deal(
            product_description=name,
            price=price,
            url=f"https://example.com/{name}",
        ),
        estimate=estimate,
        discount=estimate - price,
    )


def scraped_deal(url: str, title: str = "Product"):
    return SimpleNamespace(
        url=url,
        title=title,
        describe=lambda: f"Title: {title}\nURL: {url}",
    )


def scanner_response(url: str):
    selection = DealSelection(
        deals=[Deal(product_description="Selected product", price=100, url=url)]
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=selection.model_dump_json()))]
    )


class DealSeenStoreTests(unittest.TestCase):
    def make_store(self):
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(f"seen_{uuid.uuid4().hex}")
        return DealSeenStore(collection), collection

    def test_canonicalizes_tracking_variants_without_dropping_product_parameters(self):
        first = "HTTPS://Example.com/product/?sku=42&utm_source=email&iref=rss#buy"
        second = "https://example.com/product?iref=homepage&sku=42"

        self.assertEqual(canonicalize_url(first), "https://example.com/product?sku=42")
        self.assertEqual(canonicalize_url(first), canonicalize_url(second))

    def test_collapses_feed_duplicates_and_persists_across_store_instances(self):
        store, collection = self.make_store()
        first = scraped_deal("https://example.com/deal?sku=42&utm_source=rss")
        duplicate = scraped_deal("https://example.com/deal?sku=42&iref=rss")

        unseen = store.unseen([first, duplicate])
        store.mark_processed(unseen, [])
        restarted_store = DealSeenStore(collection)

        self.assertEqual(unseen, [first])
        self.assertEqual(restarted_store.unseen([duplicate]), [])
        self.assertEqual(collection.count(), 1)


class ScannerDeduplicationTests(unittest.TestCase):
    def make_store(self):
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(f"seen_{uuid.uuid4().hex}")
        return DealSeenStore(collection), collection

    def test_successful_scan_marks_selected_and_rejected_candidates(self):
        store, collection = self.make_store()
        selected = scraped_deal("https://example.com/selected?iref=rss", "Selected")
        rejected = scraped_deal("https://example.com/rejected", "Rejected")
        agent = ScannerAgent(seen_store=store)

        with patch("agents.scanner_agent.ScrapedDeal.fetch", return_value=[selected, rejected]), patch(
            "agents.scanner_agent.completion",
            return_value=scanner_response(selected.url),
        ) as completion:
            first = agent.scan(memory=[])
            second = agent.scan(memory=[])

        records = collection.get(include=["metadatas"])["metadatas"]
        statuses = {record["status"] for record in records}
        self.assertEqual(len(first.deals), 1)
        self.assertIsNone(second)
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(statuses, {"selected", "rejected"})

    def test_failed_scanner_call_does_not_poison_retry(self):
        store, collection = self.make_store()
        candidate = scraped_deal("https://example.com/retry", "Retry")
        agent = ScannerAgent(seen_store=store)

        with patch("agents.scanner_agent.ScrapedDeal.fetch", return_value=[candidate]), patch(
            "agents.scanner_agent.completion",
            side_effect=RuntimeError("temporary model failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "temporary model failure"):
                agent.scan(memory=[])

        self.assertEqual(collection.count(), 0)

        with patch("agents.scanner_agent.ScrapedDeal.fetch", return_value=[candidate]), patch(
            "agents.scanner_agent.completion",
            return_value=scanner_response(candidate.url),
        ) as completion:
            result = agent.scan(memory=[])

        self.assertEqual(len(result.deals), 1)
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(collection.count(), 1)


class SpecialistAgentTests(unittest.TestCase):
    def test_fine_tuned_prompt_matches_training_format(self):
        generator = Mock(return_value="1,249.50")

        result = fine_tuned_price("  A high-end laptop  ", generator=generator)

        self.assertEqual(result, 1249.50)
        generator.assert_called_once_with(
            "What does this cost to the nearest dollar?\n\n"
            "A high-end laptop\n\nPrice is $"
        )
        self.assertEqual(
            build_prompt("A high-end laptop"), generator.call_args.args[0]
        )

    def test_uses_fine_tuned_qwen_when_selected(self):
        predictor = Mock(return_value=799.0)
        agent = SpecialistAgent(backend="finetuned", price_fn=predictor)

        result = agent.price("A high-end laptop")

        self.assertEqual(result, 799.0)
        predictor.assert_called_once_with("A high-end laptop")

    def test_supports_local_ollama_fallback(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="$1,249.50"))]
        )
        completion = Mock(return_value=response)
        agent = SpecialistAgent(
            backend="ollama",
            model_name="qwen3.6:latest",
            api_base="http://localhost:11434",
            completion_fn=completion,
        )

        result = agent.price("A high-end laptop")

        self.assertEqual(result, 1249.50)
        self.assertEqual(completion.call_args.kwargs["model"], "ollama/qwen3.6:latest")
        self.assertEqual(
            completion.call_args.kwargs["api_base"], "http://localhost:11434"
        )
        self.assertFalse(completion.call_args.kwargs["think"])

    def test_rejects_non_numeric_model_output(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="unknown"))]
        )
        agent = SpecialistAgent(
            backend="ollama", completion_fn=Mock(return_value=response)
        )

        with self.assertRaisesRegex(ValueError, "no numeric price"):
            agent.price("An ambiguous product")


class PlanningProgressTests(unittest.TestCase):
    def test_streams_each_successful_estimate_and_survives_one_failure(self):
        deals = [
            Deal(product_description="one", price=100, url="https://example.com/one"),
            Deal(product_description="broken", price=100, url="https://example.com/broken"),
            Deal(product_description="three", price=100, url="https://example.com/three"),
        ]
        planner = PlanningAgent.__new__(PlanningAgent)
        planner.scanner = Mock(scan=Mock(return_value=DealSelection(deals=deals)))
        planner.messenger = Mock()

        def estimate(description):
            if description == "broken":
                raise RuntimeError("model failure")
            return {"one": 180, "three": 240}[description]

        planner.ensemble = Mock(price=Mock(side_effect=estimate))
        snapshots = []

        result = planner.plan(
            memory=[],
            on_opportunity=lambda _item, current: snapshots.append(current),
        )

        self.assertEqual([len(snapshot) for snapshot in snapshots], [1, 2])
        self.assertEqual(result.deal.product_description, "three")
        planner.messenger.alert.assert_called_once_with(result)


class FrameworkPersistenceTests(unittest.TestCase):
    def make_framework(self, memory_path: Path) -> DealAgentFramework:
        with patch("deal_agent_framework.chromadb.PersistentClient") as client:
            client.return_value.get_or_create_collection.return_value = object()
            return DealAgentFramework(db_path=memory_path.parent / "db", memory_path=memory_path)

    def test_loads_saved_rows_and_writes_valid_deduplicated_memory(self):
        saved = opportunity("saved", 100, 180)
        current = opportunity("current", 75, 160)

        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.json"
            memory_path.write_text(
                json.dumps([saved.model_dump()]),
                encoding="utf-8",
            )
            framework = self.make_framework(memory_path)
            framework.planner = Mock()

            def plan(memory, on_opportunity):
                on_opportunity(current, [current])
                return current

            framework.planner.plan.side_effect = plan
            snapshots = []
            displayed = framework.run(on_progress=lambda rows: snapshots.append(rows))

            persisted = json.loads(memory_path.read_text(encoding="utf-8"))

        self.assertEqual(len(framework.memory), 2)
        self.assertEqual(len(displayed), 2)
        self.assertTrue(any(len(snapshot) == 2 for snapshot in snapshots))
        self.assertEqual(len(persisted), 2)
        self.assertFalse(memory_path.with_suffix(".json.tmp").exists())


class UiWorkerTests(unittest.TestCase):
    def test_table_contract_contains_saved_opportunity(self):
        rows = table_for([opportunity("saved", 100, 180)])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1:4], ["$100.00", "$180.00", "$80.00"])

    def test_worker_emits_error_terminal_event_instead_of_hanging(self):
        framework = Mock()
        framework.run.side_effect = RuntimeError("pricing exploded")
        framework.display_opportunities.return_value = []
        results = queue.Queue()

        run_framework_worker(framework, results)
        event = results.get_nowait()

        self.assertEqual(event["type"], "error")
        self.assertIn("pricing exploded", event["message"])
        self.assertEqual(event["table"], [])


if __name__ == "__main__":
    unittest.main()
