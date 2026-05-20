"""
Benchmark suite — compare model quality, latency, and cost across providers.

Runs standardized test prompts through all configured providers and generates
a comparative report with rankings and recommendations.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import BaseProvider, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkPrompt:
    """A single benchmark test case."""
    id: str
    category: str
    prompt: str
    expected_contains: Optional[str] = None
    max_tokens: int = 1024
    weight: float = 1.0


@dataclass
class BenchmarkResult:
    """Result of running a single prompt against a single provider."""
    prompt_id: str
    provider: str
    model: str
    response: str
    latency_ms: float
    tokens_used: int
    cost_usd: float
    quality_score: float = 0.0  # 0-1
    passed: bool = True


@dataclass
class ProviderReport:
    """Aggregated benchmark report for a single provider."""
    provider: str
    model: str
    total_prompts: int
    passed: int
    failed: int
    avg_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    avg_quality_score: float
    results: list[BenchmarkResult] = field(default_factory=list)


# Default benchmark prompts covering key use cases
DEFAULT_PROMPTS = [
    BenchmarkPrompt(
        id="code-01",
        category="code",
        prompt="Write a Python function that implements binary search on a sorted array. Include type hints and docstring.",
        expected_contains="def binary_search",
        max_tokens=512,
    ),
    BenchmarkPrompt(
        id="code-02",
        category="code",
        prompt="Write a SQL query to find the top 5 customers by total order amount from tables: customers(id, name), orders(id, customer_id, amount).",
        expected_contains="SELECT",
        max_tokens=256,
    ),
    BenchmarkPrompt(
        id="reasoning-01",
        category="reasoning",
        prompt="A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Explain your reasoning step by step.",
        expected_contains="9",
        max_tokens=256,
    ),
    BenchmarkPrompt(
        id="reasoning-02",
        category="reasoning",
        prompt="Compare the trade-offs between microservices and monolithic architecture for a startup with 5 engineers. Give a structured analysis.",
        max_tokens=512,
    ),
    BenchmarkPrompt(
        id="math-01",
        category="math",
        prompt="Solve: If f(x) = 3x^2 + 2x - 5, find f'(x) and evaluate f'(2). Show your work.",
        expected_contains="8",
        max_tokens=256,
    ),
    BenchmarkPrompt(
        id="creative-01",
        category="creative",
        prompt="Write a haiku about artificial intelligence.",
        max_tokens=128,
    ),
    BenchmarkPrompt(
        id="analysis-01",
        category="analysis",
        prompt="Summarize the key differences between REST and GraphQL APIs in a structured comparison table format.",
        max_tokens=512,
    ),
    BenchmarkPrompt(
        id="chat-01",
        category="chat",
        prompt="What are the main benefits of using Python for data science? Keep it concise, 3-4 bullet points.",
        expected_contains="python",
        max_tokens=256,
    ),
]


class BenchmarkSuite:
    """
    Multi-provider benchmark suite.

    Runs a set of prompts across all configured providers, measures
    latency/cost/quality, and generates comparative reports.
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        prompts: list[BenchmarkPrompt] = None,
    ):
        self.providers = providers
        self.prompts = prompts or DEFAULT_PROMPTS
        self.results: list[BenchmarkResult] = []

    async def run(
        self,
        concurrency: int = 3,
        quality_evaluator=None,
    ) -> dict[str, ProviderReport]:
        """
        Run all prompts against all providers.

        Args:
            concurrency: Max concurrent requests per provider
            quality_evaluator: Optional async fn(prompt, response) -> float (0-1)

        Returns:
            Dict mapping provider name to ProviderReport
        """
        logger.info(
            f"Starting benchmark: {len(self.prompts)} prompts x {len(self.providers)} providers"
        )
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []

        for provider_name, provider in self.providers.items():
            for prompt in self.prompts:
                tasks.append(
                    self._run_single(semaphore, provider_name, provider, prompt, quality_evaluator)
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.results = [r for r in results if isinstance(r, BenchmarkResult)]

        return self._generate_reports()

    async def _run_single(
        self,
        semaphore: asyncio.Semaphore,
        provider_name: str,
        provider: BaseProvider,
        prompt: BenchmarkPrompt,
        quality_evaluator,
    ) -> BenchmarkResult:
        """Run a single prompt against a single provider."""
        async with semaphore:
            request = CompletionRequest(
                messages=[{"role": "user", "content": prompt.prompt}],
                model=provider.config.default_model,
                max_tokens=prompt.max_tokens,
            )

            try:
                response = await provider.complete(request)

                # Basic quality check
                quality = 0.5
                if prompt.expected_contains:
                    if prompt.expected_contains.lower() in response.content.lower():
                        quality = 1.0
                    else:
                        quality = 0.2

                if quality_evaluator:
                    quality = await quality_evaluator(prompt, response)

                return BenchmarkResult(
                    prompt_id=prompt.id,
                    provider=provider_name,
                    model=response.model,
                    response=response.content[:200],
                    latency_ms=response.latency_ms,
                    tokens_used=response.total_tokens,
                    cost_usd=response.cost_usd,
                    quality_score=quality,
                    passed=quality >= 0.5,
                )
            except Exception as e:
                logger.error(f"Benchmark failed for {provider_name}/{prompt.id}: {e}")
                return BenchmarkResult(
                    prompt_id=prompt.id,
                    provider=provider_name,
                    model=provider.config.default_model,
                    response=f"ERROR: {e}",
                    latency_ms=0,
                    tokens_used=0,
                    cost_usd=0,
                    quality_score=0,
                    passed=False,
                )

    def _generate_reports(self) -> dict[str, ProviderReport]:
        """Aggregate results into per-provider reports."""
        reports = {}
        for provider_name in self.providers:
            provider_results = [r for r in self.results if r.provider == provider_name]
            if not provider_results:
                continue

            passed = sum(1 for r in provider_results if r.passed)
            total_lat = sum(r.latency_ms for r in provider_results)
            total_tok = sum(r.tokens_used for r in provider_results)
            total_cost = sum(r.cost_usd for r in provider_results)
            avg_quality = sum(r.quality_score for r in provider_results) / len(provider_results)

            reports[provider_name] = ProviderReport(
                provider=provider_name,
                model=provider_results[0].model,
                total_prompts=len(provider_results),
                passed=passed,
                failed=len(provider_results) - passed,
                avg_latency_ms=total_lat / len(provider_results),
                total_tokens=total_tok,
                total_cost_usd=round(total_cost, 6),
                avg_quality_score=round(avg_quality, 3),
                results=provider_results,
            )
        return reports

    def print_report(self, reports: dict[str, ProviderReport]):
        """Print a formatted benchmark report to stdout."""
        print("\n" + "=" * 70)
        print("  AgentFlow Benchmark Report")
        print("=" * 70)

        # Sort by quality score (descending)
        sorted_providers = sorted(
            reports.values(), key=lambda r: r.avg_quality_score, reverse=True
        )

        print(f"\n{'Provider':<15} {'Model':<25} {'Pass':>5} {'Latency':>10} {'Cost':>10} {'Quality':>8}")
        print("-" * 70)
        for r in sorted_providers:
            print(
                f"{r.provider:<15} {r.model:<25} {r.passed}/{r.total_prompts:<3} "
                f"{r.avg_latency_ms:>8.0f}ms {r.total_cost_usd:>$9.4f} {r.avg_quality_score:>7.3f}"
            )

        # Winner
        if sorted_providers:
            winner = sorted_providers[0]
            print(f"\n🏆 Best overall: {winner.provider}/{winner.model} (quality: {winner.avg_quality_score:.3f})")

        print("=" * 70 + "\n")
