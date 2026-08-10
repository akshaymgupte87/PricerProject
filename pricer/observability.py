"""Local latency, success, and quality measurements."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from pricer.intelligence_store import IntelligenceStore


class Observability:
    def __init__(self, store: IntelligenceStore):
        self.store = store

    @contextmanager
    def trace(self, operation: str, **tags) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        except Exception as error:
            self.store.metric(f"{operation}.errors", 1, error=type(error).__name__, **tags)
            raise
        else:
            self.store.metric(f"{operation}.success", 1, **tags)
        finally:
            self.store.metric(
                f"{operation}.latency_ms", (perf_counter() - started) * 1000, **tags
            )
