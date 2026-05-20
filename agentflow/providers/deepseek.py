"""
DeepSeek provider — supports DeepSeek Chat and DeepSeek Reasoner.
"""

import json
import logging
import time
from typing import AsyncIterator

import httpx

from .base import BaseProvider, CompletionRequest, CompletionResponse, ProviderConfig

logger = logging.getLogger(__name__)

DEEPSEEK_PRICING = {
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}


class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider with reasoning chain support."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url or "https://api.deepseek.com/v1",
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
        async with self._semaphore:
            resp = await self.client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000
        choice = data["choices"][0]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        reasoning_tokens = usage.get("reasoning_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        model = data.get("model", request.model)
        pricing = DEEPSEEK_PRICING.get(model, {"input": 0.14, "output": 0.28})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        self._record_metrics(latency_ms, total_tokens)
        reasoning = choice["message"].get("reasoning_content")

        return CompletionResponse(
            content=choice["message"].get("content", ""),
            model=model,
            provider="deepseek",
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            reasoning=reasoning,
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
            {"id": k, "name": k.replace("-", " ").title(), "context": 65536}
            for k in DEEPSEEK_PRICING
        ]
