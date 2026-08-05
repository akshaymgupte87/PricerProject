"""Portable, resource-aware loading of Amazon product metadata.

The loader has two distinct stages:
1. Download (or reuse) a normalized Hugging Face dataset.
2. Parse and filter records with a bounded process pool.

Worker selection is adaptive so the same code runs efficiently on a laptop,
a workstation, or a constrained CI environment without exhausting RAM or file
handles.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from math import ceil

from datasets import Dataset
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from pricer.parser import parse, AMAZON_METADATA_FEATURES, _amazon_metadata_rows

# Each task sent to a worker contains this many materialized records. This is
# large enough to amortize inter-process overhead but small enough to bound RAM.
CHUNK_SIZE = 1000

# Do not start a process unless there is enough work to justify its startup and
# serialization costs. For example, 94,327 rows request ceil(94,327/25,000) = 4.
ROWS_PER_WORKER = 25_000

# Conservative planning estimates—not actual allocations. They keep the loader
# from creating more workers than a low-memory machine can support.
MEMORY_PER_WORKER = 512 * 1024**2
MEMORY_RESERVE = 2 * 1024**3


def recommended_worker_count(record_count: int, requested: int | None = None) -> int:
    """Return a safe worker count based on CPU, RAM, and amount of work.

    A caller may request a worker count, but it is still capped by detected
    resources. With ``requested=None``, the function selects the count fully
    automatically.
    """
    # process_cpu_count() respects container limits and CPU affinity. Fall back
    # to os.cpu_count() on platforms where it is unavailable.
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    logical_cpus = process_cpu_count() or os.cpu_count() or 1
    physical_cpus = logical_cpus
    memory_limit = logical_cpus

    try:
        import psutil

        # This parsing work is CPU-bound. Physical cores generally outperform
        # using every hyperthread because extra processes add IPC contention.
        physical_cpus = psutil.cpu_count(logical=False) or logical_cpus

        # Preserve memory for the notebook and operating system, then budget a
        # conservative 512 MiB for each worker process.
        available_memory = psutil.virtual_memory().available
        usable_memory = max(available_memory - MEMORY_RESERVE, available_memory // 2)
        memory_limit = max(1, usable_memory // MEMORY_PER_WORKER)
    except ImportError:
        # psutil ships with Jupyter, but CPU-based selection remains portable
        # when this module is used without Jupyter installed.
        pass

    # All three limits must be respected; the smallest one is the safe ceiling.
    resource_limit = max(1, min(logical_cpus, physical_cpus, memory_limit))

    # Small datasets often finish faster with fewer processes because starting
    # and feeding a process pool has a measurable fixed cost.
    work_limit = max(1, ceil(record_count / ROWS_PER_WORKER))
    target = work_limit if requested is None else max(1, requested)
    return min(target, resource_limit)


class ItemLoader:
    def __init__(self, category: str):
        self.category = category
        self.dataset: Dataset | None = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Yield plain Python rows in bounded chunks.

        Materializing here is important: sending Arrow Dataset slices directly
        to child processes would make them inherit/open memory-mapped files,
        which previously contributed to the "Too many open files" failure.
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size))).to_list()

    def load_in_parallel(self, workers):
        """
        Parse chunks concurrently while keeping queued work bounded.
        """
        results = []
        chunk_count = (len(self.dataset) + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Avoid process startup entirely on single-core/low-memory machines or
        # when the dataset is too small to benefit from multiprocessing.
        if workers == 1:
            for batch in tqdm(
                map(self.from_chunk, self.chunk_generator()),
                total=chunk_count,
                desc=f"Processing {self.category}",
                unit="chunk",
                dynamic_ncols=True,
            ):
                results.extend(batch)
            return results

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(
                pool.map(
                    self.from_chunk,
                    self.chunk_generator(),
                    # At most two pending chunks per worker. Without this bound,
                    # a large generator can queue thousands of chunks and consume
                    # excessive memory, pipes, and OS handles.
                    buffersize=max(workers * 2, 1),
                ),
                total=chunk_count,
                desc=f"Processing {self.category}",
                unit="chunk",
                dynamic_ncols=True,
            ):
                results.extend(batch)
        return results

    def load(self, workers=None):
        """
        Download, normalize, parse, and filter one product category.

        Leave ``workers`` as None for portable automatic tuning. Supplying a
        value requests that many processes, subject to the same safety caps.
        """
        start = datetime.now()
        print(f"  Phase 1/2: Downloading or opening cached {self.category} data...", flush=True)
        # Hugging Face/Xet widget callbacks run on background threads. In a
        # notebook, those callbacks create ZMQ event pipes and previously caused
        # handle exhaustion. Our own text progress remains visible in phase 2.
        disable_progress_bars()

        # The repository's legacy dataset script is unsupported by datasets 5.x,
        # so the project generator reads JSONL and applies a fixed schema.
        self.dataset =  Dataset.from_generator(
            _amazon_metadata_rows,
            gen_kwargs={"category": self.category},
            features=AMAZON_METADATA_FEATURES,
        )
        # Select workers only after loading because dataset size is one of the
        # inputs to the adaptive policy.
        workers = recommended_worker_count(len(self.dataset), requested=workers)
        print(
            f"  Phase 2/2: Processing {len(self.dataset):,} records with "
            f"{workers} adaptive worker{'s' if workers != 1 else ''}...",
            flush=True,
        )
        try:
            results = self.load_in_parallel(workers)
        finally:
            # Release the Arrow dataset and its memory map before the next
            # category is loaded.
            self.dataset = None
        finish = datetime.now()
        print(
            f"  Completed {self.category}: {len(results):,} usable items "
            f"in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
