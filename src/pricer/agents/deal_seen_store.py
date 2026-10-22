"""Persistent deduplication for deals observed in external feeds."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_PARAMETERS = {
    "campaign",
    "fbclid",
    "gclid",
    "iref",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
}


def canonicalize_url(url: str) -> str:
    """Normalize a deal URL while retaining product-identifying parameters."""
    value = url.strip()
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("URL has an invalid port") from error
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        hostname = f"{hostname}:{port}"

    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query = []
    for key, query_value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_PARAMETERS:
            continue
        query.append((key, query_value))

    return urlunsplit((scheme, hostname, path, urlencode(sorted(query)), ""))


def deal_id(url: str) -> str:
    """Return the stable Chroma ID for a canonical deal URL."""
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DealSeenStore:
    """Track every feed candidate after a successful scanner-model call.

    This collection is metadata-only. A one-dimensional constant embedding keeps
    Chroma from invoking its default embedding model for records that are never
    used in vector search.
    """

    def __init__(self, collection):
        self.collection = collection

    def unseen(self, deals):
        """Return unseen deals in stable order, collapsing feed-local duplicates."""
        unique_by_id = {}
        for deal in deals:
            unique_by_id.setdefault(deal_id(deal.url), deal)

        ids = list(unique_by_id)
        if not ids:
            return []

        records = self.collection.get(ids=ids, include=["metadatas"])
        terminal = {
            identifier
            for identifier, metadata in zip(records["ids"], records["metadatas"])
            if (metadata or {}).get("status") in {"rejected", "priced"}
        }
        return [deal for identifier, deal in unique_by_id.items() if identifier not in terminal]

    def mark_processed(self, deals, selected_deals) -> None:
        """Persist all candidates once their scanner response has been validated."""
        selected_urls = {canonicalize_url(deal.url) for deal in selected_deals}
        unique_by_id = {}
        for deal in deals:
            unique_by_id.setdefault(deal_id(deal.url), deal)

        if not unique_by_id:
            return

        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ids = []
        embeddings = []
        metadatas = []
        for identifier, deal in unique_by_id.items():
            canonical_url = canonicalize_url(deal.url)
            selected = canonical_url in selected_urls
            ids.append(identifier)
            embeddings.append([0.0])
            metadatas.append(
                {
                    "canonical_url": canonical_url,
                    "original_url": deal.url,
                    "title": str(getattr(deal, "title", ""))[:500],
                    "selected": selected,
                    "status": "pending" if selected else "rejected",
                    "first_seen_at": observed_at,
                }
            )

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def mark_priced(self, deal) -> None:
        """Mark a selected deal terminal after its priced opportunity is persisted."""
        identifier = deal_id(deal.url)
        records = self.collection.get(ids=[identifier], include=["metadatas"])
        if not records["ids"]:
            return
        metadata = dict(records["metadatas"][0] or {})
        metadata["status"] = "priced"
        self.collection.update(ids=[identifier], metadatas=[metadata])
