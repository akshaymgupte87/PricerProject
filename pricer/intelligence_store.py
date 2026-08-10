"""Durable local storage for opportunities, feedback, history, and telemetry."""

from __future__ import annotations

import json
import sqlite3
import threading
import math
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.deal_seen_store import canonicalize_url
from agents.deals import Opportunity
from agents.guardrails import validate_price


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IntelligenceStore:
    """A small SQLite control plane shared by the UI, API, graph, and MCP server."""

    def __init__(self, path: str | Path = "artifacts/pricer_intelligence.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS opportunities (
            url TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            decision TEXT NOT NULL,
            estimated_price REAL,
            corrected_price REAL,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            observed_price REAL NOT NULL,
            estimated_value REAL,
            source TEXT NOT NULL DEFAULT 'deal_scan',
            observed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS price_history_url_time
            ON price_history(url, observed_at);
        CREATE TABLE IF NOT EXISTS saved_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            query TEXT NOT NULL,
            filters TEXT NOT NULL DEFAULT '{}',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            tags TEXT NOT NULL DEFAULT '{}',
            recorded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            run_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
        with self._lock, closing(self.connect()) as connection:
            connection.executescript(schema)
            # Existing databases predate estimate snapshots. Keeping the estimate
            # beside the decision makes historical evaluation stable even when an
            # opportunity is rescanned and its payload is updated later.
            feedback_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(feedback)")
            }
            if "estimated_price" not in feedback_columns:
                connection.execute("ALTER TABLE feedback ADD COLUMN estimated_price REAL")
            connection.commit()

    def upsert_opportunity(self, opportunity: Opportunity) -> None:
        now = utc_now()
        if opportunity.created_at is None:
            opportunity.created_at = now
        payload = opportunity.model_dump_json()
        url = canonicalize_url(opportunity.deal.url)
        with self._lock, closing(self.connect()) as connection:
            connection.execute(
                """
                INSERT INTO opportunities(url, payload, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    payload=excluded.payload, status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (url, payload, opportunity.status, opportunity.created_at, now),
            )
            connection.execute(
                """INSERT INTO price_history
                   (url, observed_price, estimated_value, source, observed_at)
                   VALUES (?, ?, ?, 'deal_scan', ?)""",
                (url, opportunity.deal.price, opportunity.estimate, now),
            )
            connection.commit()

    def list_opportunities(self, status: str | None = None) -> list[Opportunity]:
        query = "SELECT payload, status FROM opportunities"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC"
        with closing(self.connect()) as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            item = Opportunity.model_validate_json(row["payload"])
            item.status = row["status"]
            result.append(item)
        return result

    def record_feedback(
        self,
        url: str,
        decision: str,
        *,
        corrected_price: float | None = None,
        reason: str = "",
    ) -> Opportunity:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if corrected_price is not None:
            corrected_price = validate_price(corrected_price)
        canonical = canonicalize_url(url)
        now = utc_now()
        with self._lock, closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM opportunities WHERE url = ?", (canonical,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown opportunity: {url}")
            opportunity = Opportunity.model_validate_json(row["payload"])
            opportunity.status = decision
            connection.execute(
                "UPDATE opportunities SET payload = ?, status = ?, updated_at = ? WHERE url = ?",
                (opportunity.model_dump_json(), decision, now, canonical),
            )
            connection.execute(
                """INSERT INTO feedback
                   (url, decision, estimated_price, corrected_price, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    canonical,
                    decision,
                    opportunity.estimate,
                    corrected_price,
                    reason.strip(),
                    now,
                ),
            )
            connection.commit()
        return opportunity

    def price_history(self, url: str, limit: int = 100) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT observed_price, estimated_value, source, observed_at
                   FROM price_history WHERE url = ?
                   ORDER BY observed_at DESC LIMIT ?""",
                (canonicalize_url(url), max(1, min(limit, 1000))),
            ).fetchall()
        return [dict(row) for row in rows]

    def price_insights(self, url: str) -> dict[str, Any]:
        """Summarize history and make a transparent buy-now/wait recommendation."""
        history = list(reversed(self.price_history(url, limit=1000)))
        if not history:
            return {"samples": 0, "recommendation": "insufficient_data"}
        prices = [float(item["observed_price"]) for item in history]
        average = sum(prices) / len(prices)
        variance = sum((price - average) ** 2 for price in prices) / len(prices)
        deviation = math.sqrt(variance)
        current = prices[-1]
        trend = 0.0
        if len(prices) > 1:
            trend = (prices[-1] - prices[0]) / (len(prices) - 1)
        latest_estimate = history[-1].get("estimated_value")
        discount_ratio = (
            (float(latest_estimate) - current) / float(latest_estimate)
            if latest_estimate and float(latest_estimate) > 0
            else 0.0
        )
        recommendation = "buy_now" if current <= average * 0.9 or discount_ratio >= 0.2 else "wait"
        return {
            "samples": len(prices),
            "current": current,
            "average": average,
            "minimum": min(prices),
            "maximum": max(prices),
            "trend_per_observation": trend,
            "anomaly_zscore": (current - average) / deviation if deviation else 0.0,
            "discount_ratio": discount_ratio,
            "recommendation": recommendation,
        }

    def evaluation_summary(self) -> dict[str, Any]:
        """Evaluate estimates against human corrections and approval decisions."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT f.decision, f.estimated_price, f.corrected_price, o.payload
                   FROM feedback f JOIN opportunities o ON o.url = f.url
                   ORDER BY f.created_at"""
            ).fetchall()
        absolute_errors = []
        signed_errors = []
        approved = 0
        for row in rows:
            approved += row["decision"] == "approved"
            if row["corrected_price"] is not None:
                estimate = row["estimated_price"]
                if estimate is None:  # Backward compatibility for pre-migration feedback.
                    estimate = Opportunity.model_validate_json(row["payload"]).estimate
                error = float(estimate) - float(row["corrected_price"])
                signed_errors.append(error)
                absolute_errors.append(abs(error))
        return {
            "feedback_count": len(rows),
            "approved_count": approved,
            "rejected_count": len(rows) - approved,
            "approval_rate": approved / len(rows) if rows else 0.0,
            "corrected_price_count": len(absolute_errors),
            "mean_absolute_error": (
                sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
            ),
            "root_mean_squared_error": (
                math.sqrt(sum(error**2 for error in signed_errors) / len(signed_errors))
                if signed_errors
                else None
            ),
            "mean_signed_error": (
                sum(signed_errors) / len(signed_errors) if signed_errors else None
            ),
        }

    def save_search(self, name: str, query: str, filters: dict[str, Any] | None = None) -> int:
        now = utc_now()
        with self._lock, closing(self.connect()) as connection:
            cursor = connection.execute(
                """INSERT INTO saved_searches(name, query, filters, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET query=excluded.query,
                       filters=excluded.filters, enabled=1""",
                (name.strip(), query.strip(), json.dumps(filters or {}), now),
            )
            connection.commit()
            return int(cursor.lastrowid or 0)

    def list_saved_searches(self) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id, name, query, filters, enabled, created_at FROM saved_searches ORDER BY name"
            ).fetchall()
        return [
            {**dict(row), "filters": json.loads(row["filters"]), "enabled": bool(row["enabled"])}
            for row in rows
        ]

    def saved_search_matches(self, identifier: int) -> list[Opportunity]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT query, filters FROM saved_searches WHERE id = ? AND enabled = 1",
                (identifier,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown saved search: {identifier}")
        terms = [term.casefold() for term in str(row["query"]).split() if term]
        filters = json.loads(row["filters"])
        matches = []
        for opportunity in self.list_opportunities():
            description = opportunity.deal.product_description.casefold()
            if terms and not all(term in description for term in terms):
                continue
            if any(
                str(getattr(opportunity.deal, key, None) or "").casefold()
                != str(expected).casefold()
                for key, expected in filters.items()
            ):
                continue
            matches.append(opportunity)
        return matches

    def metric(self, name: str, value: float, **tags: Any) -> None:
        with self._lock, closing(self.connect()) as connection:
            connection.execute(
                "INSERT INTO metrics(name, value, tags, recorded_at) VALUES (?, ?, ?, ?)",
                (name, float(value), json.dumps(tags), utc_now()),
            )
            connection.commit()

    def metric_summary(self) -> list[dict[str, Any]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT name, COUNT(*) AS samples, AVG(value) AS average,
                          MIN(value) AS minimum, MAX(value) AS maximum
                   FROM metrics GROUP BY name ORDER BY name"""
            ).fetchall()
        return [dict(row) for row in rows]

    def save_checkpoint(self, run_id: str, state: dict[str, Any]) -> None:
        with self._lock, closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO checkpoints(run_id, state, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET state=excluded.state,
                       updated_at=excluded.updated_at""",
                (run_id, json.dumps(state), utc_now()),
            )
            connection.commit()

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT state FROM checkpoints WHERE run_id = ?", (run_id,)
            ).fetchone()
        return json.loads(row["state"]) if row else None
