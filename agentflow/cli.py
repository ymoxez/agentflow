"""AgentFlow CLI — command-line interface for the inference gateway."""

import argparse
import asyncio
import json
import sys
import yaml

from agentflow.gateway.gateway import Gateway, GatewayConfig
from agentflow.providers.base import CompletionRequest, ProviderConfig
from agentflow.utils.logger import setup_logging


def load_config(path: str = "config/default.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def build_gateway(config: dict) -> Gateway:
    """Build a Gateway from configuration."""
    gw_config = GatewayConfig(
        cache_enabled=config.get("gateway", {}).get("cache_enabled", True),
    )
    gw = Gateway(gw_config)

    for provider_cfg in config.get("providers", []):
        if not provider_cfg.get("enabled", True):
            continue
        name = provider_cfg["name"]
        pc = ProviderConfig(
            name=name,
            api_key=provider_cfg.get("api_key", ""),
            base_url=provider_cfg.get("base_url", ""),
            default_model=provider_cfg.get("default_model", ""),
        )
        # Dynamic import based on provider name
        try:
            if name == "xiaomi":
                from agentflow.providers.xiaomi import XiaomiProvider
                gw.register_provider(name, XiaomiProvider(pc))
            elif name == "anthropic":
                from agentflow.providers.anthropic import AnthropicProvider
                gw.register_provider(name, AnthropicProvider(pc))
            elif name == "openai":
                from agentflow.providers.openai import OpenAIProvider
                gw.register_provider(name, OpenAIProvider(pc))
            elif name == "deepseek":
                from agentflow.providers.deepseek import DeepSeekProvider
                gw.register_provider(name, DeepSeekProvider(pc))
        except Exception as e:
            print(f"Warning: Failed to load provider {name}: {e}")

    return gw


async def cmd_chat(args, gw: Gateway):
    """Interactive chat mode."""
    print("AgentFlow Chat (type 'quit' to exit, 'stats' for usage)")
    print(f"Model: {args.model or 'auto-routing'}")
    print("-" * 40)

    history = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            stats = gw.get_stats()
            print(json.dumps(stats, indent=2, default=str))
            continue

        history.append({"role": "user", "content": user_input})

        request = CompletionRequest(
            messages=history,
            model=args.model or "auto",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        response = await gw.complete(request)
        print(f"\n[{response.provider}/{response.model}] {response.content}")
        print(f"  ({response.total_tokens} tokens, ${response.cost_usd:.4f}, {response.latency_ms:.0f}ms)")

        history.append({"role": "assistant", "content": response.content})


async def cmd_benchmark(args, gw: Gateway):
    """Run benchmark suite."""
    from agentflow.benchmark.suite import BenchmarkSuite
    suite = BenchmarkSuite(gw.providers)
    reports = await suite.run(concurrency=args.concurrency)
    suite.print_report(reports)


async def cmd_stats(args, gw: Gateway):
    """Show usage statistics."""
    stats = gw.tracker.get_summary(hours=args.hours)
    print(json.dumps(stats, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description="AgentFlow — AI Inference Gateway")
    parser.add_argument("--config", "-c", default="config/default.yaml", help="Config file")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="command")

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--model", "-m", help="Model to use (default: auto-route)")
    p_chat.add_argument("--max-tokens", type=int, default=4096)
    p_chat.add_argument("--temperature", type=float, default=0.7)

    p_bench = sub.add_parser("benchmark", help="Run benchmark suite")
    p_bench.add_argument("--concurrency", type=int, default=3)

    p_stats = sub.add_parser("stats", help="Show usage stats")
    p_stats.add_argument("--hours", type=int, default=24)

    sub.add_parser("serve", help="Start API server")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return

    config = load_config(args.config)
    gw = build_gateway(config)

    if args.command == "chat":
        asyncio.run(cmd_chat(args, gw))
    elif args.command == "benchmark":
        asyncio.run(cmd_benchmark(args, gw))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args, gw))
    elif args.command == "serve":
        try:
            import uvicorn
            from agentflow.api.server import app, set_gateway
            set_gateway(gw)
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except ImportError:
            print("Install uvicorn: pip install uvicorn")


if __name__ == "__main__":
    main()
