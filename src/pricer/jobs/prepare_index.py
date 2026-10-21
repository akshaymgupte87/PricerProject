"""Create the Chroma product collection and BM25 sidecar from the public dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from pricer.agents.retrieval import SQLiteBM25Index
from pricer.config import Settings
from pricer.data.items import Item


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ed-donner/items_full")
    parser.add_argument("--max-items", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chroma-path", type=Path, default=Settings.from_env().vectorstore_path)
    parser.add_argument("--bm25-path", type=Path, default=Settings.from_env().bm25_path)
    args = parser.parse_args()

    if args.max_items < 1:
        raise ValueError("--max-items must be at least 1")
    stream = load_dataset(args.dataset, split="train", streaming=True)
    items = []
    for row in stream:
        item = Item.model_validate(row)
        if item.summary and item.price >= 0:
            items.append(item)
            if len(items) >= args.max_items:
                break
    if not items:
        raise ValueError("The selected dataset contains no products with summaries and prices.")

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection("products")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        documents = [item.summary for item in batch]
        embeddings = encoder.encode(documents, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[str(item.id if item.id is not None else start + offset) for offset, item in enumerate(batch)],
            documents=documents,
            embeddings=embeddings,
            metadatas=[{"price": float(item.price), "category": item.category} for item in batch],
        )
        print(f"Indexed {min(start + len(batch), len(items)):,}/{len(items):,} products")

    count = SQLiteBM25Index(args.bm25_path).rebuild(collection, batch_size=args.batch_size)
    print(f"Prepared {count:,} products in {args.chroma_path} and {args.bm25_path}")


if __name__ == "__main__":
    main()
