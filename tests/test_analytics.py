"""Tests for the usage tracker."""

import os
import tempfile
import time
import pytest
from agentflow.analytics.tracker import UsageTracker, UsageRecord


@pytest.fixture
def tracker():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    t = UsageTracker(path)
    yield t
    t.close()
    os.unlink(path)


class TestUsageTracker:
    def test_record_usage(self, tracker):
        tracker.record(UsageRecord(
            timestamp=time.time(),
            provider="xiaomi",
            model="mimo-v2.5-pro",
            category="code",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cost_usd=0.01,
            latency_ms=500,
        ))
        summary = tracker.get_summary(hours=1)
        assert summary["total_requests"] == 1
        assert summary["total_tokens"] == 300

    def test_budget_setting(self, tracker):
        tracker.set_budget("test", 10.0, "daily")
        alerts = tracker.check_budget("test")
        assert len(alerts) == 1
        assert alerts[0].limit_usd == 10.0
        assert not alerts[0].exceeded

    def test_budget_exceeded(self, tracker):
        tracker.set_budget("small", 0.001, "daily")
        tracker.record(UsageRecord(
            timestamp=time.time(),
            provider="test",
            model="test",
            category="test",
            prompt_tokens=100,
            completion_tokens=100,
            total_tokens=200,
            cost_usd=0.01,
            latency_ms=100,
        ))
        alerts = tracker.check_budget("small")
        assert alerts[0].exceeded

    def test_optimization_tips(self, tracker):
        tips = tracker.get_optimization_tips()
        assert len(tips) > 0

    def test_summary_by_provider(self, tracker):
        tracker.record(UsageRecord(
            timestamp=time.time(),
            provider="xiaomi",
            model="mimo-v2.5-pro",
            category="code",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
            cost_usd=0.01,
            latency_ms=500,
        ))
        tracker.record(UsageRecord(
            timestamp=time.time(),
            provider="anthropic",
            model="claude-sonnet-4-0",
            category="code",
            prompt_tokens=150,
            completion_tokens=250,
            total_tokens=400,
            cost_usd=0.05,
            latency_ms=800,
        ))
        summary = tracker.get_summary(hours=1)
        assert len(summary["by_provider"]) == 2
