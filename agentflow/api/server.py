"""
REST API server — exposes AgentFlow as an OpenAI-compatible API endpoint.

Enables seamless integration with Claude Code, Cursor, OpenClaw,
and other tools that consume OpenAI-format APIs.
"""

import asyncio
import json
import logging
import time
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..gateway.gateway import Gateway
from ..providers.base import CompletionRequest

logger = logging.getLogger(__name__)

if HAS_FASTAPI:
    app = FastAPI(
        title="AgentFlow API",
        description="OpenAI-compatible multi-model inference gateway",
        version="1.0.0",
    )

    _gateway: Optional[Gateway] = None

    class ChatCompletionRequest(BaseModel):
        model: str = "auto"
        messages: list[dict]
        temperature: float = 0.7
        max_tokens: int = 4096
        stream: bool = False
        tools: Optional[list[dict]] = None
        tool_choice: Optional[str] = None
        stop: Optional[list[str]] = None

    def set_gateway(gateway: Gateway):
        """Set the gateway instance for the API server."""
        global _gateway
        _gateway = gateway

    @app.get("/v1/models")
    async def list_models():
        """List available models across all providers."""
        models = []
        for name, provider in _gateway.providers.items():
            provider_models = await provider.list_models()
            for m in provider_models:
                models.append({
                    "id": m["id"],
                    "object": "model",
                    "owned_by": name,
                    "context_window": m.get("context", 128000),
                })
        # Add auto-routing pseudo-model
        models.insert(0, {
            "id": "auto",
            "object": "model",
            "owned_by": "agentflow",
            "context_window": 0,
        })
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""
        if not _gateway:
            raise HTTPException(500, "Gateway not initialized")

        request = CompletionRequest(
            messages=req.messages,
            model=req.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=req.stream,
            tools=req.tools,
            tool_choice=req.tool_choice,
            stop=req.stop,
        )

        if req.stream:
            return StreamingResponse(
                _stream_response(request),
                media_type="text/event-stream",
            )

        response = await _gateway.complete(request)

        return {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.content,
                },
                "finish_reason": response.finish_reason,
            }],
            "usage": {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
            "x_agentflow": {
                "provider": response.provider,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
            },
        }

    async def _stream_response(request: CompletionRequest):
        """Generate SSE stream for streaming responses."""
        async for chunk in _gateway.stream(request):
            data = json.dumps({
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": chunk}}],
            })
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    @app.get("/v1/stats")
    async def get_stats():
        """Get gateway usage statistics."""
        return _gateway.get_stats()

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        providers = {}
        for name, provider in _gateway.providers.items():
            try:
                h = await provider.health_check()
                providers[name] = h.status.value
            except:
                providers[name] = "unknown"
        return {"status": "ok", "providers": providers}

else:
    app = None
    logger.warning("FastAPI not installed. API server unavailable. Install with: pip install fastapi uvicorn")
