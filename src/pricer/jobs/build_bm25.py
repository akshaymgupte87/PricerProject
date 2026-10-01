"""Build the persistent BM25 sidecar used by the hybrid product retriever."""

from __future__ import annotations

import argparse
from pathlib import Path
from pricer.config import Settings, runtime_home

import chromadb

from pricer.agents.retrieval import SQLiteBM25Index


PROJECT_ROOT = runtime_home()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Settings.from_env().vectorstore_path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Settings.from_env().bm25_path,
    )
    parser.add_argument("--collection", default="products")
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()

    collection = chromadb.PersistentClient(path=str(args.chroma_path)).get_collection(
        args.collection
    )
    count = SQLiteBM25Index(args.output).rebuild(
        collection, batch_size=args.batch_size
    )
    print(f"Indexed {count:,} documents in {args.output}")


if __name__ == "__main__":
    main()
