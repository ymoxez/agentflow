"""Tests for the model router."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.router.model_router import ModelRouter, RoutingPolicy
from agentflow.providers.base import CompletionRequest, CompletionResponse, ProviderStatus, ProviderHealth


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.config.default_model = "test-model"
    provider.health = ProviderHealth(status=ProviderStatus.HEALTHY)
    provider.complete = AsyncMock(return_value=CompletionResponse(
        content="test response",
        model="test-model",
        provider="test-provider",
        total_tokens=100,
        cost_usd=0.001,
        latency_ms=500,
    ))
    return provider


@pytest.fixture
def router(mock_provider):
    return ModelRouter({"test": mock_provider})


class TestModelRouter:
    @pytest.mark.asyncio
    async def test_route_returns_response(self, router):
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="auto",
        )
        response, decision = await router.route(request)
        assert response.content == "test response"
        assert decision.selected_provider == "test"

    @pytest.mark.asyncio
    async def test_route_force_provider(self, router):
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="auto",
        )
        policy = RoutingPolicy(force_provider="test", force_model="test-model")
        response, decision = await router.route(request, policy)
        assert decision.reason == "Forced by policy"

    @pytest.mark.asyncio
    async def test_usage_stats(self, router):
        request = CompletionRequest(
            messages=[{"role": "user", "content": "Write a Python function"}],
            model="auto",
        )
        await router.route(request)
        stats = router.get_usage_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 100
