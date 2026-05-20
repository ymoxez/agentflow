# ⚡ AgentFlow

**AI Inference Gateway & Multi-Model Benchmarking Platform**

AgentFlow is a unified API gateway that intelligently routes LLM requests across multiple providers (Xiaomi MiMo, Anthropic Claude, OpenAI GPT, DeepSeek), with automatic model selection based on task type, built-in benchmarking, cost optimization, and OpenAI-compatible API.

## Why AgentFlow?

Running AI-powered workflows at scale means juggling multiple providers, models, and billing systems. AgentFlow eliminates that complexity:

- **One API, many models** — Single OpenAI-compatible endpoint that routes to the best model for each task
- **Smart routing** — Automatically classifies prompts (code, reasoning, creative, math) and selects the optimal model
- **Cost optimization** — Track spending across providers, set budgets, get optimization tips
- **Benchmark everything** — Compare model quality, latency, and cost side-by-side
- **Pipeline chains** — Chain multiple models for complex multi-step workflows

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       AgentFlow 1.0                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│   │   REST API  │   │    CLI      │   │   Python    │      │
│   │ (OpenAI fmt)│   │  (chat/     │   │    SDK      │      │
│   │             │   │  benchmark) │   │             │      │
│   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│          │                 │                  │              │
│   ┌──────┴─────────────────┴──────────────────┴──────┐      │
│   │              Unified Gateway                      │      │
│   │   • Request routing    • Response caching         │      │
│   │   • Budget enforcement • Usage analytics          │      │
│   └──────────────────────┬───────────────────────────┘      │
│                          │                                   │
│   ┌──────────────────────┴───────────────────────────┐      │
│   │              Smart Model Router                   │      │
│   │   • Task classification (7 categories)            │      │
│   │   • Health-aware failover                         │      │
│   │   • Cost/quality trade-off optimization           │      │
│   └──────────────────────┬───────────────────────────┘      │
│                          │                                   │
│   ┌──────────┬───────────┼───────────┬──────────────┐       │
│   │          │           │           │              │       │
│  ┌┴───┐  ┌──┴───┐  ┌───┴──┐  ┌────┴────┐  ┌─────┐│       │
│  │MiMo│  │Claude│  │ GPT  │  │DeepSeek│  │ ... ││       │
│  └────┘  └──────┘  └──────┘  └────────┘  └─────┘│       │
│   Xiaomi  Anthropic  OpenAI   DeepSeek  Extensible│       │
│                                                      │       │
│   ┌──────────────────────────────────────────────┐  │       │
│   │           Analytics Engine                    │  │       │
│   │   • Per-request metrics (tokens, cost, lat)   │  │       │
│   │   • Budget tracking with alerts               │  │       │
│   │   • Optimization recommendations              │  │       │
│   │   • Provider comparison reports               │  │       │
│   └──────────────────────────────────────────────┘  │       │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -e "."

# Set API keys
export XIAOMI_API_KEY="your-mimo-key"
export ANTHROPIC_API_KEY="your-claude-key"
export OPENAI_API_KEY="your-openai-key"

# Interactive chat with auto-routing
agentflow chat

# Use a specific model
agentflow chat --model mimo-v2.5-pro

# Run benchmarks across all providers
agentflow benchmark

# View usage stats
agentflow stats --hours 48

# Start API server (OpenAI-compatible)
agentflow serve
```

## Smart Routing

AgentFlow classifies prompts into 7 categories and routes to the optimal model:

| Category | Best Model | Why |
|----------|-----------|-----|
| **Code** | Claude Sonnet | Best at code generation, debugging, refactoring |
| **Reasoning** | MiMo V2.5 Pro | Strong chain-of-thought, analysis |
| **Creative** | GPT-4o | Natural language, storytelling |
| **Math** | DeepSeek Reasoner | Dedicated reasoning with step-by-step |
| **Analysis** | MiMo V2.5 Pro | Data extraction, summarization |
| **Translation** | DeepSeek Chat | Cost-effective multilingual |
| **Vision** | GPT-4o | Image understanding |

```python
from agentflow.router.task_classifier import classify_task

result = classify_task("Write a Python function to parse JSON")
# → category: CODE, model: claude-sonnet-4-0, confidence: 0.9

result = classify_task("Prove that √2 is irrational")
# → category: MATH, model: deepseek-reasoner, confidence: 0.8
```

## Python SDK

```python
import asyncio
from agentflow.gateway.gateway import Gateway, GatewayConfig
from agentflow.providers.base import CompletionRequest, ProviderConfig
from agentflow.providers.xiaomi import XiaomiProvider
from agentflow.providers.anthropic import AnthropicProvider

async def main():
    gw = Gateway(GatewayConfig())

    # Register providers
    gw.register_provider("xiaomi", XiaomiProvider(ProviderConfig(
        name="xiaomi",
        api_key="your-key",
        base_url="https://api.xiaomimimo.com/v1",
        default_model="mimo-v2.5-pro",
    )))
    gw.register_provider("anthropic", AnthropicProvider(ProviderConfig(
        name="anthropic",
        api_key="your-key",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-0",
    )))

    # Auto-routed request
    response = await gw.complete(CompletionRequest(
        messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    ))
    print(f"[{response.provider}/{response.model}] {response.content}")
    print(f"Tokens: {response.total_tokens}, Cost: ${response.cost_usd:.4f}")

asyncio.run(main())
```

## Multi-Model Pipelines

Chain multiple models for complex workflows:

```python
from agentflow.pipeline.chain import create_research_pipeline

pipeline = create_research_pipeline(providers)
result = await pipeline.execute("Research the current state of AI agents in 2026")

# Step 1: GPT-4o researches → Step 2: MiMo analyzes → Step 3: Claude summarizes
print(result.final_output)
print(f"Total: {result.total_tokens} tokens, ${result.total_cost_usd}")
```

## Benchmarking

Compare providers side-by-side:

```python
from agentflow.benchmark.suite import BenchmarkSuite

suite = BenchmarkSuite(providers)
reports = await suite.run(concurrency=5)
suite.print_report(reports)
```

```
======================================================================
  AgentFlow Benchmark Report
======================================================================

Provider        Model                    Pass   Latency       Cost  Quality
----------------------------------------------------------------------
anthropic       claude-sonnet-4-0         8/8     1200ms   $0.0234    0.925
xiaomi          mimo-v2.5-pro             7/8      800ms   $0.0045    0.875
openai          gpt-4o                    8/8     1500ms   $0.0312    0.900
deepseek        deepseek-chat             6/8      600ms   $0.0021    0.750

🏆 Best overall: anthropic/claude-sonnet-4-0 (quality: 0.925)
======================================================================
```

## API Server

Start an OpenAI-compatible API server:

```bash
agentflow serve --port 8000
```

Then use it with any OpenAI-compatible tool:

```bash
# Works with Claude Code, Cursor, OpenClaw, etc.
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=agentflow

# The "auto" model triggers smart routing
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Cost Optimization

```python
# Set budgets
tracker.set_budget("daily", 10.0, "daily")
tracker.set_budget("monthly", 200.0, "monthly")

# Get optimization tips
tips = tracker.get_optimization_tips()
# → "Consider using gpt-4o-mini for simpler tasks — you used gpt-4o 87 times this week."
```

## Token Consumption Profile

AgentFlow is designed for high-throughput AI workflows. Typical daily usage:

| Operation | Model | Tokens/Day |
|-----------|-------|-----------|
| Code generation | Claude Sonnet | ~1M |
| Reasoning tasks | MiMo V2.5 Pro | ~800K |
| Creative/content | GPT-4o | ~500K |
| Quick queries | MiMo Lite / DeepSeek | ~300K |
| Benchmarks | All providers | ~200K |
| **Total** | | **~2-3M/day** |

## Project Structure

```
agentflow/
├── gateway/           # Unified entry point
│   └── gateway.py     # Routing, caching, budget enforcement
├── providers/         # LLM provider implementations
│   ├── base.py        # Abstract provider interface
│   ├── xiaomi.py      # Xiaomi MiMo (V2.5 Pro/Lite/Vision)
│   ├── anthropic.py   # Anthropic Claude (Opus/Sonnet/Haiku)
│   ├── openai.py      # OpenAI GPT (4o/4.1/o3/o4-mini)
│   └── deepseek.py    # DeepSeek (Chat/Reasoner)
├── router/            # Intelligent model routing
│   ├── task_classifier.py  # 7-category prompt classification
│   └── model_router.py     # Health-aware routing with fallback
├── benchmark/         # Multi-provider benchmark suite
│   └── suite.py       # 8 benchmark prompts, comparative reports
├── pipeline/          # Multi-model workflow chains
│   └── chain.py       # Sequential/parallel pipeline execution
├── analytics/         # Usage tracking & cost optimization
│   └── tracker.py     # SQLite metrics, budgets, recommendations
├── api/               # OpenAI-compatible REST API
│   └── server.py      # FastAPI server with streaming
├── utils/             # Logging and helpers
│   └── logger.py
└── cli.py             # Command-line interface
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check .
```

## License

MIT License — see [LICENSE](LICENSE) for details.
