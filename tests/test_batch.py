import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pricer.batch import Batch


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class BatchConcurrencyTests(unittest.TestCase):
    def test_run_local_processes_requests_concurrently_and_writes_valid_jsonl(self):
        items = [
            SimpleNamespace(id=index, full=f"item {index}", summary=None)
            for index in range(6)
        ]
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def fake_chat(messages):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return messages[-1]["content"]

        with tempfile.TemporaryDirectory() as temp_dir, working_directory(temp_dir):
            batch = Batch(items, 0, len(items), lite=True)
            with patch("pricer.batch.ollama_chat", side_effect=fake_chat):
                batch.run_local(workers=3)

            output = Path("lite/output/0_6.jsonl")
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertGreater(maximum_active, 1)
        self.assertEqual({record["custom_id"] for record in records}, {str(i) for i in range(6)})
        self.assertEqual(len(records), 6)


if __name__ == "__main__":
    unittest.main()
