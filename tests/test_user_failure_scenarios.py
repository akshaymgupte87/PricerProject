"""Regression tests for user-facing failure scenarios found in the code review."""

import tempfile
import threading
import unittest
import sys
import requests
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

from pricer.agents.deals import Deal, Opportunity, ScrapedDeal
from pricer.application.framework import DealAgentFramework
from pricer.interfaces.dashboard import review_risk
from pricer.persistence.intelligence_store import IntelligenceStore
from pricer.agents.scanner_agent import ScannerAgent
from pricer.agents.specialist_agent import SpecialistAgent
from pricer.interfaces.schemas import SavedSearchRequest
from pricer.evaluation.evaluator import Tester


class DealPageFailureTests(unittest.TestCase):
    def test_changed_detail_markup_falls_back_to_rss_summary(self):
        response = Mock(status_code=200, content=b"<html><body>New page layout</body></html>")
        with patch("pricer.agents.deals.requests.get", return_value=response) as get:
            deal = ScrapedDeal(
                {
                    "title": "Laptop deal",
                    "summary": "<p>Discounted laptop details</p>",
                    "links": [{"href": "https://example.com/deal"}],
                }
            )

        self.assertEqual(deal.details, "Discounted laptop details")
        get.assert_called_once_with("https://example.com/deal", timeout=(5, 15), allow_redirects=False)

    def test_one_unreadable_feed_item_does_not_discard_other_items(self):
        entries = [
            {"title": "broken", "summary": "x", "links": []},
            {
                "title": "good",
                "summary": "Useful summary",
                "links": [{"href": "https://example.com/good"}],
            },
        ]
        with patch(
            "pricer.agents.deals.feedparser.parse",
            side_effect=[Mock(entries=entries), Mock(entries=[]), Mock(entries=[])],
        ), patch(
            "pricer.agents.deals.requests.get",
            return_value=Mock(status_code=200, content=b"<div class='content-section'>Full details</div>"),
        ), patch("pricer.agents.deals.time.sleep"):
            deals = ScrapedDeal.fetch()

        self.assertEqual([deal.title for deal in deals], ["good"])

    def test_unavailable_feed_does_not_discard_later_feeds(self):
        first = Mock(status_code=503, content=b"bad")
        first.raise_for_status.side_effect = RuntimeError("503")
        good = Mock(status_code=200, content=b"empty")
        with patch("pricer.agents.deals.requests.get", side_effect=[first, good, good]), patch(
            "pricer.agents.deals.feedparser.parse", return_value=Mock(entries=[])
        ) as parse:
            self.assertEqual(ScrapedDeal.fetch(), [])
        self.assertEqual(parse.call_count, 2)

    def test_redirect_is_followed_only_after_validating_destination(self):
        redirect = Mock(status_code=302, headers={"Location": "https://example.org/item"})
        final = Mock(status_code=200, content=b"ok")
        with patch("pricer.agents.deals.requests.get", side_effect=[redirect, final]) as get:
            from pricer.agents.deals import get_public_url
            self.assertIs(get_public_url("https://example.com/item"), final)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args.args[0], "https://example.org/item")

    def test_redirect_to_private_host_is_rejected_before_second_request(self):
        redirect = Mock(status_code=302, headers={"Location": "http://127.0.0.1/admin"})
        with patch("pricer.agents.deals.requests.get", return_value=redirect) as get:
            from pricer.agents.deals import get_public_url
            with self.assertRaisesRegex(ValueError, "public IP"):
                get_public_url("https://example.com/item")
        get.assert_called_once()

    def test_redirect_without_location_and_redirect_loop_fail_clearly(self):
        from pricer.agents.deals import get_public_url
        with patch("pricer.agents.deals.requests.get", return_value=Mock(status_code=302, headers={})), self.assertRaisesRegex(ValueError, "Location"):
            get_public_url("https://example.com/item")
        redirect = Mock(status_code=302, headers={"Location": "/again"})
        with patch("pricer.agents.deals.requests.get", return_value=redirect) as get, self.assertRaises(requests.TooManyRedirects):
            get_public_url("https://example.com/item")
        self.assertEqual(get.call_count, 6)

    def test_rejects_private_and_malformed_deal_urls(self):
        for url in ("http://127.0.0.1/a", "http://localhost/a", "https://example.com:bad/a", "https://user:pass@example.com/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Deal(product_description="Laptop", price=10, url=url)

    def test_scraper_does_not_fetch_private_targets(self):
        with patch("pricer.agents.deals.requests.get") as get:
            with self.assertRaisesRegex(ValueError, "public"):
                ScrapedDeal({"title": "bad", "summary": "x", "links": [{"href": "http://127.0.0.1/"}]})
        get.assert_not_called()


class ScannerOutputTests(unittest.TestCase):
    def test_model_cannot_introduce_unseen_urls(self):
        scraped = Mock(url="https://example.com/candidate?utm_source=feed", describe=Mock(return_value="candidate"))
        accepted = Deal(product_description="Laptop", price=100, url="https://example.com/candidate")
        invented = Deal(product_description="Other", price=100, url="https://example.com/invented")
        response = Mock(choices=[Mock(message=Mock(content=__import__("json").dumps({"deals": [accepted.model_dump(), invented.model_dump()]})))])
        agent = ScannerAgent()
        agent.fetch_deals = Mock(return_value=[scraped])
        with patch("pricer.agents.scanner_agent.completion", return_value=response):
            result = agent.scan(memory=[])
        self.assertEqual([item.url for item in result.deals], [scraped.url])


class InputValidationTests(unittest.TestCase):
    def test_saved_search_rejects_whitespace_only_name_or_query(self):
        for name, query in (("   ", "laptop"), ("watch", "  ")):
            with self.subTest(name=name, query=query), self.assertRaises(ValueError):
                SavedSearchRequest(name=name, query=query)

    def test_evaluator_rejects_request_larger_than_available_labels(self):
        tester = Tester(lambda _item: 1, [{"prompt": "item", "completion": 1}], size=2, workers=1)
        with self.assertRaisesRegex(ValueError, "dataset has only 1"):
            tester.run()

    def test_url_canonicalization_reports_invalid_port(self):
        from pricer.agents.deal_seen_store import canonicalize_url
        with self.assertRaisesRegex(ValueError, "invalid port"):
            canonicalize_url("https://example.com:broken/item")

    def test_neural_evaluate_rejects_empty_validation_loader(self):
        import torch
        from pricer.jobs.train_neural import evaluate
        with self.assertRaisesRegex(ValueError, "no examples"):
            evaluate(Mock(eval=Mock()), [], torch.device("cpu"), 0.0, 1.0)

    def test_neural_training_rejects_empty_splits_before_training(self):
        from pricer.jobs import train_neural
        with patch.object(sys, "argv", ["train_neural"]), patch.object(
            train_neural.Item, "from_hub", return_value=([], [], [])
        ), self.assertRaisesRegex(ValueError, "training split"):
            train_neural.main()

    def test_price_parser_rejects_multiple_numbers_instead_of_using_product_model_number(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            SpecialistAgent._extract_price("RTX 4090 laptop, price $1,200")


class AlertRetryTests(unittest.TestCase):
    def test_approval_alert_retries_after_delivery_failure_then_sends_once(self):
        item = Opportunity(deal=Deal(product_description="Laptop", price=100, url="https://example.com/alert"), estimate=200, discount=100)
        with tempfile.TemporaryDirectory() as directory:
            framework = DealAgentFramework.__new__(DealAgentFramework)
            framework.store = IntelligenceStore(Path(directory) / "state.sqlite3")
            framework.memory_path = Path(directory) / "memory.json"
            framework.store.upsert_opportunity(item)
            framework.memory = []
            framework.current_run = []
            with patch("pricer.application.framework.MessagingAgent") as messaging:
                messaging.return_value.alert.side_effect = [RuntimeError("delivery failed"), None]
                with self.assertRaisesRegex(RuntimeError, "delivery failed"):
                    framework.submit_feedback(item.deal.url, "approved")
                framework.submit_feedback(item.deal.url, "approved")
                framework.submit_feedback(item.deal.url, "approved")
            self.assertEqual(messaging.return_value.alert.call_count, 2)

    def test_alert_claim_is_exclusive_across_store_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            stores = [IntelligenceStore(path), IntelligenceStore(path)]
            barrier = threading.Barrier(2)
            results = []
            def claim(store):
                barrier.wait()
                results.append(store.claim_alert_delivery("https://example.com/race"))
            threads = [threading.Thread(target=claim, args=(store,)) for store in stores]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(sorted(results), [False, True])

    def test_abandoned_pending_alert_claim_can_be_retried_after_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "state.sqlite3")
            self.assertTrue(store.claim_alert_delivery("https://example.com/abandoned"))
            with closing(store.connect()) as connection:
                connection.execute(
                    "UPDATE alert_deliveries SET updated_at=?",
                    ("2000-01-01T00:00:00+00:00",),
                )
                connection.commit()
            self.assertTrue(store.claim_alert_delivery("https://example.com/abandoned"))


class PricingReviewRiskTests(unittest.TestCase):
    def test_high_heuristic_confidence_without_comparables_still_requires_high_risk_review(self):
        opportunity = Opportunity(
            deal=Deal(
                product_description="Laptop",
                price=100,
                url="https://example.com/laptop",
            ),
            estimate=200,
            discount=100,
            model_estimates={"a": 200, "b": 201, "c": 199},
            confidence=0.99,
            evidence=[],
        )

        level, reasons = review_risk(opportunity)

        self.assertEqual(level, "High")
        self.assertIn("no retrieved comparables", reasons)


class DashboardTriggerTests(unittest.TestCase):
    def test_overlapping_timer_or_manual_trigger_is_skipped_while_scan_is_running(self):
        framework = DealAgentFramework.__new__(DealAgentFramework)
        framework.store = Mock()
        framework.store.list_opportunities.return_value = []
        framework.memory = []
        framework.current_run = []
        framework._run_lock = threading.Lock()
        framework.planner = Mock()
        framework._run_lock.acquire()
        try:
            self.assertEqual(framework.run(), [])
        finally:
            framework._run_lock.release()

        framework.planner.plan.assert_not_called()


class SavedSearchScopeTests(unittest.TestCase):
    def test_saved_search_only_matches_persisted_opportunities_when_queried(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IntelligenceStore(Path(directory) / "state.sqlite3")
            search_id = store.save_search("Laptop watch", "gaming laptop")
            self.assertEqual(store.saved_search_matches(search_id), [])

            opportunity = Opportunity(
                deal=Deal(
                    product_description="Gaming laptop with OLED display",
                    price=900,
                    url="https://example.com/oled-laptop",
                ),
                estimate=1200,
                discount=300,
            )
            store.upsert_opportunity(opportunity)

            matches = store.saved_search_matches(search_id)

        self.assertEqual([item.deal.url for item in matches], [opportunity.deal.url])


class DurableStatePrecedenceTests(unittest.TestCase):
    def test_sqlite_snapshot_wins_over_stale_memory_copy(self):
        persisted = Opportunity(
            deal=Deal(
                product_description="Laptop",
                price=100,
                url="https://example.com/laptop",
            ),
            estimate=200,
            discount=100,
            status="approved",
        )
        stale_memory = persisted.model_copy(deep=True)
        stale_memory.status = "pending"
        stale_memory.estimate = 250

        (merged,) = DealAgentFramework._merge_opportunities([persisted], [stale_memory])

        self.assertEqual(merged.status, "approved")
        self.assertEqual(merged.estimate, 200)


if __name__ == "__main__":
    unittest.main()
