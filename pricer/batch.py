"""Generate product summaries with a local Ollama model.

Unlike Groq's hosted batch API, Ollama serves requests synchronously. This
module preserves the project's batch files and checkpoints every response so a
stopped run can resume without repeating completed model calls.
"""

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

MODEL = "qwen3.6:latest"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
STATE_FILE = Path("batches.pkl")
REQUEST_TIMEOUT_SECONDS = 600


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer setting without making module import fragile."""
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


OLLAMA_CONCURRENCY = _positive_int_from_env("OLLAMA_CONCURRENCY", 4)

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


def ollama_chat_response(messages: list[dict[str, str]]) -> dict:
    """Return the complete response from one local Ollama chat request."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # Qwen can emit a separate reasoning trace. Summarization only needs the
        # final answer, so disabling thinking reduces latency and output size.
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
    }
    request = Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`."
        ) from exc

    if not isinstance(result.get("message", {}).get("content"), str):
        raise RuntimeError(f"Unexpected Ollama response: {result}")
    return result


def ollama_chat(messages: list[dict[str, str]]) -> str:
    """Return only the generated text from a local Ollama chat request."""
    result = ollama_chat_response(messages)
    return result["message"]["content"].strip()


class Batch:
    BATCH_SIZE = 1_000
    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches_folder = folder / BATCHES_FOLDER
        self.output_folder = folder / OUTPUT_FOLDER
        self.batches_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def messages_for(item):
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.full},
        ]

    def make_jsonl(self, item):
        """Create an auditable local request record (not a hosted API request)."""
        return json.dumps(
            {
                "custom_id": str(item.id),
                "model": MODEL,
                "messages": self.messages_for(item),
            },
            ensure_ascii=False,
        )

    def make_file(self):
        batch_file = self.batches_folder / self.filename
        with batch_file.open("w", encoding="utf-8") as file:
            for item in self.items[self.start : self.end]:
                file.write(self.make_jsonl(item))
                file.write("\n")

    def _completed_ids(self) -> set[str]:
        """Return IDs already checkpointed in this batch's output file."""
        output_file = self.output_folder / self.filename
        if not output_file.exists():
            return set()

        completed = set()
        with output_file.open(encoding="utf-8") as file:
            for line in file:
                try:
                    completed.add(str(json.loads(line)["custom_id"]))
                except (json.JSONDecodeError, KeyError):
                    # Ignore a partial final line left by an interrupted write.
                    continue
        return completed

    def run_local(self, workers: int | None = None):
        """Generate summaries concurrently and checkpoint them from one writer."""
        output_file = self.output_folder / self.filename
        completed_ids = self._completed_ids()
        batch_items = self.items[self.start : self.end]
        workers = OLLAMA_CONCURRENCY if workers is None else max(1, workers)

        # If the previous process stopped during its final write, start on a new
        # line. The malformed partial line is ignored and that item is retried.
        if output_file.exists() and output_file.stat().st_size:
            with output_file.open("rb+") as raw_file:
                raw_file.seek(-1, 2)
                if raw_file.read(1) != b"\n":
                    raw_file.write(b"\n")

        pending_items = [
            item for item in batch_items if str(item.id) not in completed_ids
        ]
        if not pending_items:
            return

        # Ollama calls are I/O-bound from Python's perspective, so threads allow
        # several requests to be in flight. Only this main thread writes the
        # checkpoint file, preventing interleaved/corrupt JSONL records.
        with (
            ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ollama"
            ) as executor,
            output_file.open("a", encoding="utf-8") as file,
        ):
            futures = {
                executor.submit(ollama_chat, self.messages_for(item)): str(item.id)
                for item in pending_items
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Summarizing {self.start:,}-{self.end:,} ({workers} workers)",
                unit="item",
                dynamic_ncols=True,
            ):
                item_id = futures[future]
                summary = future.result()
                # Keep the prior response shape so apply_output remains simple
                # and existing output-processing code stays compatible.
                result = {
                    "custom_id": item_id,
                    "response": {
                        "body": {"choices": [{"message": {"content": summary}}]}
                    },
                }
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                completed_ids.add(item_id)

    def is_ready(self):
        expected = self.end - self.start
        return len(self._completed_ids()) >= expected

    def apply_output(self):
        output_file = self.output_folder / self.filename
        with output_file.open(encoding="utf-8") as file:
            for line in file:
                try:
                    json_line = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[item_id].summary = summary
        self.done = self.is_ready()

    @classmethod
    def create(cls, items, lite):
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            cls.batches.append(cls(items, start, end, lite))
        print(f"Created {len(cls.batches)} local Ollama batches using {MODEL}")

    @classmethod
    def run(cls, workers: int | None = None):
        """Run unfinished batches with concurrent requests inside each batch."""
        for batch in tqdm(cls.batches, desc="Local batches", unit="batch"):
            if not batch.done:
                batch.make_file()
                batch.run_local(workers=workers)
                batch.apply_output()
        print(f"Completed {sum(batch.done for batch in cls.batches)} local batches")

    @classmethod
    def fetch(cls):
        """Apply any completed local output files after loading saved state."""
        for batch in cls.batches:
            if not batch.done and batch.is_ready():
                batch.apply_output()
        finished = sum(batch.done for batch in cls.batches)
        print(f"Finished {finished} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with STATE_FILE.open("wb") as file:
            pickle.dump(cls.batches, file)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with STATE_FILE.open("rb") as file:
            cls.batches = pickle.load(file)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
