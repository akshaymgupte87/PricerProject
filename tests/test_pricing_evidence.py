"""Protect the benchmark's metric and leakage contracts."""
import json
from pathlib import Path
import tempfile
import unittest

from pricer.jobs.pricing_evidence import evaluate


class EvidenceTests(unittest.TestCase):
    def test_constant_labels_have_zero_error_and_reproducible_split(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            rows = [{'id': str(i), 'text': f'appliance model variant{i}', 'price': 20.0} for i in range(50)]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            report = evaluate(path)
            self.assertEqual(report['train_size'] + report['test_size'], 50)
            for result in report['results'].values():
                self.assertAlmostEqual(result['mae'], 0)
                self.assertEqual(result['failure_rate'], 0)
            self.assertEqual(len(report['predictions']), 2 * report['test_size'])
            path.write_text('\n'.join(json.dumps(r) for r in reversed(rows)), encoding='utf-8')
            again = evaluate(path)
            self.assertEqual({p['id'] for p in report['predictions']}, {p['id'] for p in again['predictions']})

    def test_duplicate_titles_are_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                {'id': 'a', 'text': 'Same  Title', 'price': 10},
                {'id': 'b', 'text': 'same title', 'price': 20}]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                evaluate(path)
