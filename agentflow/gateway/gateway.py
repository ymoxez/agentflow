"""
Unified gateway — single entry point for all LLM interactions.

Provides a clean API that handles routing, fallback, caching,
budget enforcement, and analytics transparently.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import BaseProvider, CompletionRequest, CompletionResponse, ProviderConfig
from ..router.model_router import ModelRouter, RoutingPolicy
from ..analytics.tracker import UsageTracker, UsageRecord

logger = logging.getLogger(__name__)


@dataclass
class GatewayConfig:
    """Configuration for the AgentFlow gateway."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    budget_alert_threshold: float = 0.8  # Alert at 80%
    default_routing_policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    analytics_db: str = "agentflow_analytics.db"


class Gateway:
    """
    Unified LLM gateway with intelligent routing, caching, and analytics.

    Usage:
        gw = Gateway()
        gw.register_provider("xiaomi", XiaomiProvider(config))
        gw.register_provider("anthropic", AnthropicProvider(config))

        response = await gw.complete(CompletionRequest(
            messages=[{"role": "user", "content": "Explain quantum computing"}],
        ))
    """

    def __init__(self, config: GatewayConfig = None):
        self.config = config or GatewayConfig()
        self.providers: dict[str, BaseProvider] = {}
        self.router = ModelRouter(self.providers)
        self.tracker = UsageTracker(self.config.analytics_db)
        self._cache: dict[str, tuple[float, CompletionResponse]] = {}
        self._cache_order: list[str] = []

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register an LLM provider."""
        self.providers[name] = provider
        self.router.providers[name] = provider
        logger.info(f"Provider registered: {name}")

    async def complete(
        self,
        request: CompletionRequest,
        policy: RoutingPolicy = None,
    ) -> CompletionResponse:
        """
        Complete a request through the gateway.

        Handles: routing, caching, budget checks, analytics.
        """
        policy = policy or self.config.default_routing_policy

        # Budget check
        alerts = self.tracker.check_budget()
        for alert in alerts:
            if alert.exceeded:
                raise RuntimeError(
                    f"Budget '{alert.budget_name}' exceeded: "
                    f"${alert.spent_usd:.2f} / ${alert.limit_usd:.2f}"
                )
            if alert.percent_used > self.config.budget_alert_threshold * 100:
                logger.warning(
                    f"Budget alert: {alert.budget_name} at {alert.percent_used:.1f}%"
                )

        # Cache check
        if self.config.cache_enabled:
            cache_key = self._cache_key(request)
            cached = self._get_cached(cache_key)
            if cached:
                logger.info("Cache hit")
                return cached

        # Route and execute
        response, decision = await self.router.route(request, policy)

        # Cache response
        if self.config.cache_enabled and response.finish_reason == "stop":
            self._put_cached(cache_key, response)

        # Track usage
        self.tracker.record(UsageRecord(
            timestamp=time.time(),
            provider=response.provider,
            model=response.model,
            category=decision.category.value,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
        ))

        return response

    async def stream(self, request: CompletionRequest, policy: RoutingPolicy = None):
        """Stream a completion response."""
        policy = policy or self.config.default_routing_policy
        # For streaming, classify and route directly
        user_msg = ""
        for msg in request.messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        from ..router.task_classifier import classify_task
        classification = classify_task(user_msg)

        provider = self.providers.get(classification.suggested_provider)
        if not provider:
            provider = next(iter(self.providers.values()))

        request.model = classification.suggested_model
        async for chunk in provider.stream(request):
            yield chunk

    def get_stats(self) -> dict:
        """Get combined gateway statistics."""
        return {
            "providers": {name: p.get_stats() if hasattr(p, "get_stats") else {}
                          for name, p in self.providers.items()},
            "routing": self.router.get_usage_stats(),
            "analytics": self.tracker.get_summary(hours=24),
            "cache_size": len(self._cache),
            "optimization_tips": self.tracker.get_optimization_tips(),
        }

    def _cache_key(self, request: CompletionRequest) -> str:
        """Generate cache key from request content."""
        content = str(sorted([
            str(msg) for msg in request.messages
        ])) + str(request.model) + str(request.temperature)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[CompletionResponse]:
        """Get from cache if not expired."""
        if key in self._cache:
            ts, resp = self._cache[key]
            if time.time() - ts < self.config.cache_ttl_seconds:
                return resp
            del self._cache[key]
        return None

    def _put_cached(self, key: str, response: CompletionResponse):
        """Add to cache with LRU eviction."""
        if len(self._cache) >= self.config.cache_max_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = (time.time(), response)
        self._cache_order.append(key)
