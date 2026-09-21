"""Offline regression checks for the extracted notebook workflows."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pricer.experiments.config import DatasetConfig
from pricer.experiments.data import deduplicate, weighted_sample, split_items, summarized_copies
from pricer.experiments.baselines import fit_feature_regression, vectorize_training, fit_text_model
from pricer.experiments.human import write_human_template, read_human_pricer


def item(index):
    return SimpleNamespace(title=f"Product {index}", full=f"Description {index}",
                           price=float(index + 1), category="Automotive" if index % 2 else "Electronics",
                           weight=float(index), summary=f"Product speaker audio quality model {index}")


class PreparationTests(unittest.TestCase):
    def test_default_dataset_names_and_sizes(self):
        self.assertEqual(DatasetConfig().repo(), "ed-donner/items_full")
        config = DatasetConfig("owner", lite=True)
        self.assertEqual(config.repo("raw"), "owner/items_raw_lite")
        self.assertEqual(sum(config.split_sizes), 22_000)

    def test_dedup_matches_two_pass_original_without_mutation(self):
        items = [item(i) for i in range(8)]
        items[3].title = items[1].title
        items[5].full = items[2].full
        expected = list(items)
        random.Random(42).shuffle(expected)
        for field in ("title", "full"):
            seen = set()
            expected = [x for x in expected if not (getattr(x, field) in seen or seen.add(getattr(x, field)))]
        before = list(items)
        self.assertEqual(deduplicate(items), expected)
        self.assertEqual(items, before)

    def test_sampling_matches_original_and_does_not_change_rng(self):
        items = [item(i) for i in range(40)]
        prices = np.array([x.price for x in items])
        weights = ((prices - prices.min()) / (prices.max() - prices.min() + 1e-9)) ** 2
        weights[np.array([x.category for x in items]) == "Automotive"] *= 0.05
        indices = np.random.RandomState(42).choice(len(items), size=15, replace=False, p=weights / weights.sum())
        expected = [items[i] for i in indices]
        random.Random(42).shuffle(expected)
        state = random.getstate()
        np_state = np.random.get_state()
        actual = weighted_sample(items, size=15)
        self.assertEqual(actual, expected)
        self.assertEqual(random.getstate(), state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])

    def test_invalid_weight_population_is_explained(self):
        with self.assertRaisesRegex(ValueError, "positive-weight"):
            weighted_sample([item(1), item(1)], size=1)

    def test_split_is_complete_and_disjoint(self):
        items = [item(i) for i in range(10)]
        parts = split_items(items, (6, 2, 2))
        self.assertEqual([len(part) for part in parts], [6, 2, 2])
        self.assertEqual([x for part in parts for x in part], items)
        with self.assertRaises(ValueError):
            split_items(items, (6, 2, 1))

    def test_export_copies_and_rejects_incomplete_summaries(self):
        from pricer.items import Item
        original = Item(title="Speaker", price=10, category="Electronics", full="raw", id="0", summary="summary")
        exported = summarized_copies(([original], [], []))[0][0]
        self.assertEqual(original.full, "raw")
        self.assertEqual(original.id, "0")
        self.assertIsNone(exported.full)
        original.summary = ""
        with self.assertRaises(ValueError):
            summarized_copies(([original], [], []))


class PredictorTests(unittest.TestCase):
    def test_predictors_own_their_fitted_state(self):
        train = [item(i) for i in range(20)]
        predictor = fit_feature_regression(train)
        before = predictor(train[2])
        other = [item(i) for i in range(20)]
        for product in other:
            product.price *= 5
        fit_feature_regression(other)
        self.assertEqual(predictor(train[2]), before)
        vectorizer, features, prices = vectorize_training(train)
        text = fit_text_model(vectorizer, features, prices)
        self.assertTrue(np.isfinite(text(train[1])))
        self.assertIsInstance(text.__name__, str)

    def test_human_csv_rejects_wrong_order(self):
        products = [item(i) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_human_template(products, path)
            self.assertEqual(read_human_pricer(products, path)(products[0]), 0)
            with self.assertRaisesRegex(ValueError, "ordering"):
                read_human_pricer(list(reversed(products)), path)

    def test_local_llm_keeps_original_request_contract(self):
        from pricer.experiments import local_llm
        with patch.object(local_llm, "call_local_qwen", return_value="$42") as call:
            self.assertEqual(local_llm.local_qwen_3_6_pricer(item(0)), "$42")
        self.assertIn(item(0).summary, call.call_args.args[0][0]["content"])

    def test_neural_smoke_is_deterministic(self):
        from pricer.experiments.neural import fit_neural_network
        products = [item(i) for i in range(12)]
        first, history = fit_neural_network(products, epochs=1, batch_size=4)
        second, _ = fit_neural_network(products, epochs=1, batch_size=4)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(first(products[0])))
        self.assertEqual(first(products[0]), second(products[0]))


if __name__ == "__main__":
    unittest.main()
