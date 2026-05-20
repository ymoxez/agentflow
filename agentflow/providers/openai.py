"""
OpenAI GPT provider — supports GPT-4o, GPT-4.1, o3, and compatible endpoints.
"""

import json
import logging
import time
from typing import AsyncIterator

import httpx

from .base import BaseProvider, CompletionRequest, CompletionResponse, ProviderConfig

logger = logging.getLogger(__name__)

OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
}


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url or "https://api.openai.com/v1",
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
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        model = data.get("model", request.model)
        pricing = OPENAI_PRICING.get(model, {"input": 2.50, "output": 10.0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        self._record_metrics(latency_ms, total_tokens)

        return CompletionResponse(
            content=choice["message"].get("content", ""),
            model=model,
            provider="openai",
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            tool_calls=choice["message"].get("tool_calls"),
            raw_response=data,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        payload = {
            "model": request.model or self.config.default_model,
            "messages": request.messages,
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
        return [
            {"id": k, "name": k.replace("-", " ").title(), "context": 128000}
            for k in OPENAI_PRICING
        ]
