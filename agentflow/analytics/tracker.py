"""
Token usage analytics and cost optimization engine.

Tracks per-request metrics, generates cost reports, identifies
optimization opportunities, and enforces budget limits.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ANALYTICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    category TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms REAL DEFAULT 0,
    quality_score REAL,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);

CREATE TABLE IF NOT EXISTS budgets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    limit_usd REAL NOT NULL,
    period TEXT NOT NULL DEFAULT 'daily',
    spent_usd REAL DEFAULT 0,
    period_start REAL NOT NULL,
    active INTEGER DEFAULT 1
);
"""


@dataclass
class UsageRecord:
    """A single API usage record."""
    timestamp: float
    provider: str
    model: str
    category: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    quality_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BudgetAlert:
    """Alert when approaching or exceeding budget."""
    budget_name: str
    limit_usd: float
    spent_usd: float
    percent_used: float
    period: str
    exceeded: bool


class UsageTracker:
    """
    Persistent usage tracking with SQLite backend.

    Features:
    - Per-request logging with full metrics
    - Budget enforcement with alerts
    - Cost optimization recommendations
    - Provider comparison reports
    - Time-series aggregation (hourly, daily, monthly)
    """

    def __init__(self, db_path: str = "agentflow_analytics.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(ANALYTICS_SCHEMA)
        self.conn.commit()

    def record(self, usage: UsageRecord) -> None:
        """Record a single API usage."""
        self.conn.execute(
            """INSERT INTO requests
            (timestamp, provider, model, category, prompt_tokens, completion_tokens,
             total_tokens, cost_usd, latency_ms, quality_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                usage.timestamp, usage.provider, usage.model, usage.category,
                usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
                usage.cost_usd, usage.latency_ms, usage.quality_score,
                json.dumps(usage.metadata),
            ),
        )
        self.conn.commit()

    def set_budget(self, name: str, limit_usd: float, period: str = "daily") -> None:
        """Set a spending budget."""
        self.conn.execute(
            """INSERT OR REPLACE INTO budgets (name, limit_usd, period, spent_usd, period_start)
            VALUES (?, ?, ?, 0, ?)""",
            (name, limit_usd, period, time.time()),
        )
        self.conn.commit()

    def check_budget(self, budget_name: str = None) -> list[BudgetAlert]:
        """Check all budgets and return alerts for those near or over limit."""
        query = "SELECT * FROM budgets WHERE active = 1"
        params = []
        if budget_name:
            query += " AND name = ?"
            params.append(budget_name)

        budgets = self.conn.execute(query, params).fetchall()
        alerts = []

        for b in budgets:
            # Calculate spent in current period
            period_seconds = {"hourly": 3600, "daily": 86400, "monthly": 2592000}
            period_start = b["period_start"]
            cutoff = time.time() - period_seconds.get(b["period"], 86400)

            row = self.conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) as total FROM requests WHERE timestamp > ?",
                (max(period_start, cutoff),),
            ).fetchone()
            spent = row["total"]

            percent = (spent / b["limit_usd"] * 100) if b["limit_usd"] > 0 else 0
            alerts.append(BudgetAlert(
                budget_name=b["name"],
                limit_usd=b["limit_usd"],
                spent_usd=round(spent, 4),
                percent_used=round(percent, 1),
                period=b["period"],
                exceeded=spent >= b["limit_usd"],
            ))

        return alerts

    def get_summary(self, hours: int = 24) -> dict:
        """Get usage summary for the last N hours."""
        cutoff = time.time() - (hours * 3600)

        total = self.conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(total_tokens), 0) as tokens, "
            "COALESCE(SUM(cost_usd), 0) as cost, COALESCE(AVG(latency_ms), 0) as latency "
            "FROM requests WHERE timestamp > ?", (cutoff,)
        ).fetchone()

        by_provider = self.conn.execute(
            "SELECT provider, COUNT(*) as cnt, SUM(total_tokens) as tokens, "
            "SUM(cost_usd) as cost, AVG(latency_ms) as latency "
            "FROM requests WHERE timestamp > ? GROUP BY provider ORDER BY cost DESC",
            (cutoff,),
        ).fetchall()

        by_model = self.conn.execute(
            "SELECT model, COUNT(*) as cnt, SUM(total_tokens) as tokens, "
            "SUM(cost_usd) as cost "
            "FROM requests WHERE timestamp > ? GROUP BY model ORDER BY tokens DESC",
            (cutoff,),
        ).fetchall()

        return {
            "period_hours": hours,
            "total_requests": total["cnt"],
            "total_tokens": total["tokens"],
            "total_cost_usd": round(total["cost"], 4),
            "avg_latency_ms": round(total["latency"], 1),
            "by_provider": [dict(r) for r in by_provider],
            "by_model": [dict(r) for r in by_model],
        }

    def get_optimization_tips(self) -> list[str]:
        """Analyze usage patterns and suggest cost optimizations."""
        tips = []
        summary = self.get_summary(hours=168)  # Last 7 days

        # Check for expensive models used for simple tasks
        by_model = {r["model"]: r for r in summary.get("by_model", [])}
        if "gpt-4o" in by_model and by_model["gpt-4o"]["cnt"] > 50:
            tips.append(
                "Consider using gpt-4o-mini for simpler tasks — you used gpt-4o "
                f"{by_model['gpt-4o']['cnt']} times in the past week."
            )

        # Check for high latency
        if summary["avg_latency_ms"] > 10000:
            tips.append(
                f"Average latency is {summary['avg_latency_ms']:.0f}ms. "
                "Consider using lighter models for interactive tasks."
            )

        # Check for single-provider dependency
        providers = summary.get("by_provider", [])
        if len(providers) == 1:
            tips.append(
                f"You only use {providers[0]['provider']}. "
                "Diversifying providers reduces risk and may optimize costs."
            )

        if not tips:
            tips.append("Usage patterns look optimized. No recommendations at this time.")

        return tips

    def close(self):
        """Close the database connection."""
        self.conn.close()
