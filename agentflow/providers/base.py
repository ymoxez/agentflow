"""
Abstract LLM provider interface with unified request/response format.

Supports: Xiaomi MiMo, Anthropic Claude, OpenAI GPT, Google Gemini,
DeepSeek, and any OpenAI-compatible endpoint.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    RATE_LIMITED = "rate_limited"


@dataclass
class CompletionRequest:
    """Unified request format across all providers."""
    messages: list[dict]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[list[dict]] = None
    tool_choice: Optional[str] = None
    system: Optional[str] = None
    stop: Optional[list[str]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Unified response format across all providers."""
    content: str
    model: str
    provider: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tool_calls: Optional[list[dict]] = None
    reasoning: Optional[str] = None
    raw_response: Optional[dict] = None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    api_key: str
    base_url: str
    default_model: str
    max_concurrent: int = 10
    timeout: float = 120.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 1000000
    enabled: bool = True


@dataclass
class ProviderHealth:
    """Health metrics for a provider."""
    status: ProviderStatus = ProviderStatus.HEALTHY
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    requests_last_minute: int = 0
    tokens_last_minute: int = 0
    last_error: Optional[str] = None
    last_check: float = 0.0


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.health = ProviderHealth()
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._request_times: list[float] = []
        self._token_counts: list[int] = []

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request and return a response."""
        ...

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Stream a completion response token by token."""
        ...

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """List available models from this provider."""
        ...

    async def health_check(self) -> ProviderHealth:
        """Check provider health with a lightweight request."""
        start = time.monotonic()
        try:
            req = CompletionRequest(
                messages=[{"role": "user", "content": "ping"}],
                model=self.config.default_model,
                max_tokens=5,
            )
            resp = await self.complete(req)
            latency = (time.monotonic() - start) * 1000
            self.health.status = ProviderStatus.HEALTHY
            self.health.avg_latency_ms = latency
            self.health.last_check = time.time()
        except Exception as e:
            self.health.status = ProviderStatus.DOWN
            self.health.last_error = str(e)
            self.health.last_check = time.time()
        return self.health

    def _record_metrics(self, latency_ms: float, tokens: int):
        """Record request metrics for health tracking."""
        self._request_times.append(latency_ms)
        self._token_counts.append(tokens)
        # Keep last 100 samples
        if len(self._request_times) > 100:
            self._request_times = self._request_times[-100:]
            self._token_counts = self._token_counts[-100:]
        if self._request_times:
            self.health.avg_latency_ms = sum(self._request_times) / len(self._request_times)
