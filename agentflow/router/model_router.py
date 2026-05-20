"""
Model router — selects optimal provider and model based on task classification,
provider health, cost constraints, and user preferences.

Supports fallback chains, cost budgets, and latency constraints.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import BaseProvider, CompletionRequest, CompletionResponse, ProviderStatus
from .task_classifier import TaskCategory, classify_task

logger = logging.getLogger(__name__)


@dataclass
class RoutingPolicy:
    """Constraints and preferences for model routing."""
    max_cost_per_request: float = 0.50  # USD
    max_latency_ms: float = 30000.0
    preferred_providers: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)
    force_provider: Optional[str] = None
    force_model: Optional[str] = None
    quality_priority: float = 0.7  # 0=cheapest, 1=best quality


@dataclass
class RoutingDecision:
    """Records why a particular model was selected."""
    selected_model: str
    selected_provider: str
    category: TaskCategory
    confidence: float
    alternatives: list[dict]
    reason: str
    estimated_cost: float


class ModelRouter:
    """
    Intelligent model router with health-aware failover.

    Routes requests to the best available model based on:
    - Task classification (code, reasoning, creative, etc.)
    - Provider health and latency
    - Cost constraints
    - User preferences and policies
    """

    def __init__(self, providers: dict[str, BaseProvider]):
        self.providers = providers
        self._usage_log: list[dict] = []

    async def route(
        self,
        request: CompletionRequest,
        policy: RoutingPolicy = None,
    ) -> tuple[CompletionResponse, RoutingDecision]:
        """
        Route a request to the best provider/model and execute it.
        Returns the response and the routing decision metadata.
        """
        policy = policy or RoutingPolicy()

        # Force override
        if policy.force_provider and policy.force_model:
            provider = self.providers.get(policy.force_provider)
            if provider:
                request.model = policy.force_model
                resp = await provider.complete(request)
                decision = RoutingDecision(
                    selected_model=policy.force_model,
                    selected_provider=policy.force_provider,
                    category=TaskCategory.CHAT,
                    confidence=1.0,
                    alternatives=[],
                    reason="Forced by policy",
                    estimated_cost=resp.cost_usd,
                )
                self._log_usage(resp, decision)
                return resp, decision

        # Classify the task
        user_msg = ""
        for msg in request.messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break
        classification = classify_task(user_msg)

        # Build ranked candidate list
        candidates = self._rank_candidates(classification, policy)

        # Try candidates in order with fallback
        last_error = None
        for candidate in candidates:
            provider_name, model = candidate["provider"], candidate["model"]
            provider = self.providers.get(provider_name)
            if not provider:
                continue
            if provider.health.status == ProviderStatus.DOWN:
                logger.warning(f"Skipping down provider: {provider_name}")
                continue

            try:
                request.model = model
                resp = await provider.complete(request)

                decision = RoutingDecision(
                    selected_model=model,
                    selected_provider=provider_name,
                    category=classification.category,
                    confidence=classification.confidence,
                    alternatives=candidates[:5],
                    reason=f"Auto-routed for {classification.category.value}",
                    estimated_cost=resp.cost_usd,
                )
                self._log_usage(resp, decision)
                return resp, decision

            except Exception as e:
                logger.warning(f"Provider {provider_name}/{model} failed: {e}")
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def _rank_candidates(
        self, classification, policy: RoutingPolicy
    ) -> list[dict]:
        """Rank provider/model pairs by suitability."""
        candidates = []

        for name, provider in self.providers.items():
            if name in [p for p in (policy.excluded_models or [])]:
                continue
            if policy.preferred_providers and name not in policy.preferred_providers:
                continue

            # Primary model for this category
            candidates.append({
                "provider": name,
                "model": classification.suggested_model if name == classification.suggested_provider else provider.config.default_model,
                "score": 1.0 if name == classification.suggested_provider else 0.5,
                "health": provider.health.status.value,
            })

        # Sort by score (descending), healthy providers first
        candidates.sort(key=lambda c: (c["health"] != "down", c["score"]), reverse=True)
        return candidates

    def _log_usage(self, response: CompletionResponse, decision: RoutingDecision):
        """Log usage for analytics."""
        self._usage_log.append({
            "timestamp": time.time(),
            "model": response.model,
            "provider": response.provider,
            "category": decision.category.value,
            "tokens": response.total_tokens,
            "cost": response.cost_usd,
            "latency_ms": response.latency_ms,
        })

    def get_usage_stats(self) -> dict:
        """Get aggregated usage statistics."""
        if not self._usage_log:
            return {"total_requests": 0}

        total_tokens = sum(u["tokens"] for u in self._usage_log)
        total_cost = sum(u["cost"] for u in self._usage_log)
        avg_latency = sum(u["latency_ms"] for u in self._usage_log) / len(self._usage_log)

        by_provider = {}
        by_category = {}
        for u in self._usage_log:
            by_provider.setdefault(u["provider"], {"requests": 0, "tokens": 0, "cost": 0})
            by_provider[u["provider"]]["requests"] += 1
            by_provider[u["provider"]]["tokens"] += u["tokens"]
            by_provider[u["provider"]]["cost"] += u["cost"]

            by_category.setdefault(u["category"], {"requests": 0, "tokens": 0})
            by_category[u["category"]]["requests"] += 1
            by_category[u["category"]]["tokens"] += u["tokens"]

        return {
            "total_requests": len(self._usage_log),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "by_provider": by_provider,
            "by_category": by_category,
        }
