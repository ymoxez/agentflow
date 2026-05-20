"""
Xiaomi MiMo provider — native integration for MiMo V2.5 series.

Supports MiMo-V2.5-Pro, MiMo-V2.5-Lite, and MiMo-V2.5-Vision.
Optimized for reasoning tasks with thinking/reasoning token extraction.
"""

import json
import logging
import time
from typing import AsyncIterator

import httpx

from .base import BaseProvider, CompletionRequest, CompletionResponse, ProviderConfig

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD)
XIAOMI_PRICING = {
    "mimo-v2.5-pro": {"input": 0.14, "output": 0.56},
    "mimo-v2.5-lite": {"input": 0.07, "output": 0.28},
    "mimo-v2.5-vision": {"input": 0.21, "output": 0.84},
}

DEFAULT_MODELS = [
    {"id": "mimo-v2.5-pro", "name": "MiMo V2.5 Pro", "context": 131072, "reasoning": True},
    {"id": "mimo-v2.5-lite", "name": "MiMo V2.5 Lite", "context": 131072, "reasoning": False},
    {"id": "mimo-v2.5-vision", "name": "MiMo V2.5 Vision", "context": 131072, "reasoning": False},
]


class XiaomiProvider(BaseProvider):
    """Xiaomi MiMo API provider with reasoning chain extraction."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()

        payload = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.stop:
            payload["stop"] = request.stop

        async with self._semaphore:
            resp = await self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000
        choice = data["choices"][0]
        usage = data.get("usage", {})

        # Extract reasoning if present
        reasoning = None
        message = choice["message"]
        if "reasoning_content" in message:
            reasoning = message["reasoning_content"]

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost
        model = data.get("model", request.model)
        pricing = XIAOMI_PRICING.get(model, {"input": 0.14, "output": 0.56})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        self._record_metrics(latency_ms, total_tokens)

        return CompletionResponse(
            content=message.get("content", ""),
            model=model,
            provider="xiaomi",
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            reasoning=reasoning,
            tool_calls=message.get("tool_calls"),
            raw_response=data,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        payload = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    async def list_models(self) -> list[dict]:
        return DEFAULT_MODELS
