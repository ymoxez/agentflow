"""
Anthropic Claude provider — supports Claude 4/3.5 series with extended thinking.
"""

import json
import logging
import time
from typing import AsyncIterator

import httpx

from .base import BaseProvider, CompletionRequest, CompletionResponse, ProviderConfig

logger = logging.getLogger(__name__)

ANTHROPIC_PRICING = {
    "claude-opus-4-0": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-0": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
}


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url or "https://api.anthropic.com/v1",
            timeout=config.timeout,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        start = time.monotonic()

        # Extract system message (Anthropic uses top-level system param)
        system = request.system
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                messages.append(msg)

        payload = {
            "model": request.model or self.config.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = request.tools

        async with self._semaphore:
            resp = await self.client.post("/messages", json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency_ms = (time.monotonic() - start) * 1000
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        model = data.get("model", request.model)
        pricing = ANTHROPIC_PRICING.get(model, {"input": 3.0, "output": 15.0})
        cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block["text"]

        self._record_metrics(latency_ms, total_tokens)

        return CompletionResponse(
            content=content,
            model=model,
            provider="anthropic",
            finish_reason=data.get("stop_reason", "end_turn"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=data,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        system = request.system
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                messages.append(msg)

        payload = {
            "model": request.model or self.config.default_model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with self.client.stream("POST", "/messages", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta["text"]

    async def list_models(self) -> list[dict]:
        return [
            {"id": "claude-opus-4-0", "name": "Claude Opus 4", "context": 200000},
            {"id": "claude-sonnet-4-0", "name": "Claude Sonnet 4", "context": 200000},
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "context": 200000},
        ]
