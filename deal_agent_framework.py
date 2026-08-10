import os
import sys
import logging
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from agents.deal_seen_store import DealSeenStore, canonicalize_url
from agents.messaging_agent import MessagingAgent
from pricer.intelligence_store import IntelligenceStore
from pricer.observability import Observability
from sklearn.manifold import TSNE
import numpy as np

load_dotenv(override=True)

# Colors for logging
BG_BLUE = "\033[44m"
WHITE = "\033[37m"
RESET = "\033[0m"

# Colors for plot
CATEGORIES = [
    "Appliances",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Electronics",
    "Musical_Instruments",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
]
COLORS = ["red", "blue", "brown", "orange", "yellow", "green", "purple", "cyan"]


def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if any(getattr(handler, "_pricer_console_handler", False) for handler in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler._pricer_console_handler = True
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


class DealAgentFramework:
    PROJECT_ROOT = Path(__file__).resolve().parent
    DB = PROJECT_ROOT / "products_vectorstore"
    MEMORY_FILENAME = PROJECT_ROOT / "memory.json"
    INTELLIGENCE_FILENAME = PROJECT_ROOT / "artifacts" / "pricer_intelligence.sqlite3"

    def __init__(
        self,
        db_path: str | Path | None = None,
        memory_path: str | Path | None = None,
        intelligence_path: str | Path | None = None,
    ):
        init_logging()
        self.db_path = Path(db_path or self.DB)
        self.memory_path = Path(memory_path or self.MEMORY_FILENAME)
        if intelligence_path is None and memory_path is not None:
            intelligence_path = self.memory_path.parent / "pricer_intelligence.sqlite3"
        self.store = IntelligenceStore(intelligence_path or self.INTELLIGENCE_FILENAME)
        self.observability = Observability(self.store)
        client = chromadb.PersistentClient(path=str(self.db_path))
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection("products")
        self.seen_store = DealSeenStore(client.get_or_create_collection("seen_deals"))
        self.planner = None
        self.current_run: List[Opportunity] = []
        self._run_lock = threading.Lock()

    def init_agents_as_needed(self):
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection, seen_store=self.seen_store)
            self.log("Agent Framework is ready")

    def read_memory(self) -> List[Opportunity]:
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text(encoding="utf-8"))
                return [Opportunity(**item) for item in data]
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                logging.exception("Could not read deal memory from %s", self.memory_path)
        return []

    def write_memory(self) -> None:
        data = [opportunity.model_dump() for opportunity in self.memory]
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.memory_path.with_suffix(self.memory_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary_path, self.memory_path)

    @classmethod
    def reset_memory(cls) -> None:
        Path(cls.MEMORY_FILENAME).write_text("[]", encoding="utf-8")

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    @staticmethod
    def _merge_opportunities(*groups: List[Opportunity]) -> List[Opportunity]:
        """Merge snapshots by URL while retaining stable display order."""
        merged: dict[str, Opportunity] = {}
        for group in groups:
            for opportunity in group:
                merged[opportunity.deal.url] = opportunity
        return list(merged.values())

    def display_opportunities(self) -> List[Opportunity]:
        persisted = self.store.list_opportunities()
        return self._merge_opportunities(persisted, self.memory, self.current_run)

    def run(
        self,
        on_progress: Callable[[List[Opportunity]], None] | None = None,
    ) -> List[Opportunity]:
        if not self._run_lock.acquire(blocking=False):
            self.log("A planning run is already active; skipping duplicate trigger")
            return self.display_opportunities()

        try:
            self.init_agents_as_needed()
            self.current_run = []
            logging.info("Kicking off Planning Agent")

            def handle_opportunity(
                _opportunity: Opportunity,
                current: List[Opportunity],
            ) -> None:
                self.store.upsert_opportunity(_opportunity)
                self.current_run = current
                if on_progress:
                    on_progress(self.display_opportunities())

            result = self.planner.plan(
                memory=self.memory,
                on_opportunity=handle_opportunity,
            )
            logging.info("Planning Agent has completed and returned: %s", result)
            if result and all(
                existing.deal.url != result.deal.url for existing in self.memory
            ):
                self.memory.append(result)
                self.write_memory()
            final = self.display_opportunities()
            if on_progress:
                on_progress(final)
            return final
        finally:
            self._run_lock.release()

    def submit_feedback(
        self,
        url: str,
        decision: str,
        *,
        corrected_price: float | None = None,
        reason: str = "",
    ) -> Opportunity:
        canonical_url = canonicalize_url(url)
        existing = next(
            (
                opportunity
                for opportunity in self.store.list_opportunities()
                if canonicalize_url(opportunity.deal.url) == canonical_url
            ),
            None,
        )
        was_approved = existing is not None and existing.status == "approved"
        updated = self.store.record_feedback(
            url, decision, corrected_price=corrected_price, reason=reason
        )
        for opportunity in [*self.memory, *self.current_run]:
            if canonicalize_url(opportunity.deal.url) == canonical_url:
                opportunity.status = decision
        self.write_memory()
        if decision == "approved" and not was_approved:
            MessagingAgent().alert(updated)
        return updated

    def price_history(self, url: str) -> list[dict]:
        return self.store.price_history(url)

    @classmethod
    def get_plot_data(cls, max_datapoints=2000):
        client = chromadb.PersistentClient(path=str(cls.DB))
        collection = client.get_or_create_collection("products")
        result = collection.get(
            include=["embeddings", "documents", "metadatas"], limit=max_datapoints
        )
        vectors = np.array(result["embeddings"])
        documents = result["documents"]
        categories = [metadata.get("category") for metadata in result["metadatas"]]
        color_by_category = dict(zip(CATEGORIES, COLORS))
        colors = [color_by_category.get(category, "gray") for category in categories]
        if len(vectors) < 4:
            raise ValueError("The vector collection needs at least 4 records for the 3D plot")
        # TSNE requires perplexity to be smaller than the number of samples.
        perplexity = min(30, len(vectors) - 1)
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            random_state=42,
            n_jobs=-1,
        )
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__ == "__main__":
    DealAgentFramework().run()
