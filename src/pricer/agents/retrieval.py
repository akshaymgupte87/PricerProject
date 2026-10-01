"""Hybrid retrieval primitives for the pricing RAG pipeline."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
QUERY_NOISE = {"a", "an", "and", "for", "of", "the", "this", "with"}


def tokenize(text: str) -> list[str]:
    """Return FTS-safe terms while preserving product-model tokens."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def expand_product_query(query: str) -> list[str]:
    """Create conservative lexical variants without inventing product facts."""
    original = " ".join((query or "").split())
    terms = tokenize(original)
    compact = " ".join(term for term in terms if term not in QUERY_NOISE)
    model_tokens: list[str] = []
    for index, term in enumerate(terms):
        if any(character.isdigit() for character in term):
            if index and terms[index - 1] not in QUERY_NOISE:
                model_tokens.append(terms[index - 1])
            model_tokens.append(term)
    model_terms = " ".join(dict.fromkeys(model_tokens))
    return list(dict.fromkeys(value for value in (original, compact, model_terms) if value))


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif str(actual).casefold() != str(expected).casefold():
            return False
    return True


@dataclass
class RetrievalCandidate:
    id: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    lexical_rank: int | None = None
    fusion_score: float = 0.0
    reranker_score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    candidates: list[RetrievalCandidate]
    used_hybrid: bool
    reranked: bool


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class CrossEncoderReranker:
    """Lazily load a cross-encoder so application startup stays inexpensive."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        if not documents:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        pairs = [(query, document) for document in documents]
        return [float(value) for value in self._model.predict(pairs)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalCandidate]],
    *,
    rank_constant: int = 60,
) -> list[RetrievalCandidate]:
    """Fuse independent rankings without comparing incompatible raw scores."""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")

    combined: dict[str, RetrievalCandidate] = {}
    for ranking in rankings:
        for rank, candidate in enumerate(ranking, start=1):
            current = combined.get(candidate.id)
            if current is None:
                current = RetrievalCandidate(
                    id=candidate.id,
                    document=candidate.document,
                    metadata=dict(candidate.metadata),
                )
                combined[candidate.id] = current
            if candidate.dense_rank is not None:
                current.dense_rank = candidate.dense_rank
            if candidate.lexical_rank is not None:
                current.lexical_rank = candidate.lexical_rank
            current.fusion_score += 1.0 / (rank_constant + rank)
    return sorted(combined.values(), key=lambda item: item.fusion_score, reverse=True)


def rerank(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    reranker: Reranker | None,
) -> list[RetrievalCandidate]:
    if reranker is None or not candidates:
        return list(candidates)
    scores = list(reranker.score(query, [item.document for item in candidates]))
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a different number of scores than candidates")
    result = list(candidates)
    for candidate, score in zip(result, scores):
        candidate.reranker_score = float(score)
    return sorted(
        result,
        key=lambda item: (item.reranker_score, item.fusion_score),
        reverse=True,
    )


class SQLiteBM25Index:
    """Persistent lexical sidecar for a Chroma collection using SQLite FTS5."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def ready(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            with closing(sqlite3.connect(self.path)) as connection:
                row = connection.execute(
                    "SELECT value FROM index_metadata WHERE key = 'complete'"
                ).fetchone()
            return bool(row and row[0] == "1")
        except sqlite3.Error:
            return False

    def search(
        self,
        query: str,
        limit: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalCandidate]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or limit <= 0 or not self.ready:
            return []
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                """
                SELECT chroma_id, document, metadata, bm25(documents) AS score
                FROM documents
                WHERE documents MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_query, limit * 5 if metadata_filter else limit),
            ).fetchall()
        candidates = [
            RetrievalCandidate(
                id=row[0],
                document=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                lexical_rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]
        filtered = [item for item in candidates if metadata_matches(item.metadata, metadata_filter)]
        for rank, item in enumerate(filtered, start=1):
            item.lexical_rank = rank
        return filtered[:limit]

    def rebuild(self, collection, *, batch_size: int = 5_000) -> int:
        """Rebuild the sidecar from Chroma in bounded batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TABLE IF EXISTS documents")
            connection.execute("DROP TABLE IF EXISTS index_metadata")
            connection.execute(
                "CREATE VIRTUAL TABLE documents USING fts5("
                "chroma_id UNINDEXED, document, metadata UNINDEXED, "
                "tokenize='unicode61')"
            )
            connection.execute(
                "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            total = int(collection.count())
            indexed = 0
            for offset in range(0, total, batch_size):
                batch = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                rows: Iterable[tuple[str, str, str]] = (
                    (
                        str(item_id),
                        document or "",
                        json.dumps(metadata or {}, separators=(",", ":")),
                    )
                    for item_id, document, metadata in zip(
                        batch.get("ids", []),
                        batch.get("documents", []),
                        batch.get("metadatas", []),
                    )
                )
                rows = list(rows)
                connection.executemany(
                    "INSERT INTO documents(chroma_id, document, metadata) VALUES (?, ?, ?)",
                    rows,
                )
                indexed += len(rows)
                connection.commit()
            connection.execute(
                "INSERT INTO index_metadata(key, value) VALUES ('complete', '1')"
            )
            connection.execute(
                "INSERT INTO index_metadata(key, value) VALUES ('document_count', ?)",
                (str(indexed),),
            )
            connection.commit()
        return indexed


def retrieval_metrics(
    retrieved_ids: Sequence[str], relevant_ids: set[str], *, k: int = 5
) -> dict[str, float]:
    """Compute objective retrieval metrics for a single labelled query."""
    if k <= 0:
        raise ValueError("k must be positive")
    selected = list(retrieved_ids[:k])
    hits = [1 if item_id in relevant_ids else 0 for item_id in selected]
    recall = sum(hits) / len(relevant_ids) if relevant_ids else 0.0
    precision = sum(hits) / k
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        "reciprocal_rank": reciprocal_rank,
        f"ndcg@{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }
