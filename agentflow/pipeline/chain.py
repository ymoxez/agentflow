"""
Multi-model pipeline — chain multiple LLM calls for complex tasks.

Supports sequential chains, parallel fan-out, and conditional branching.
Each step can use a different model/provider optimized for its task.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..providers.base import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in a multi-model pipeline."""
    name: str
    provider: str
    model: str
    system_prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    transform: Optional[Callable] = None  # Transform output before passing to next step
    condition: Optional[Callable] = None  # Skip step if condition returns False


@dataclass
class PipelineResult:
    """Result of executing a full pipeline."""
    steps: list[dict]
    final_output: str
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    success: bool
    error: Optional[str] = None


class Pipeline:
    """
    Multi-model pipeline executor.

    Chain multiple LLM calls where each step's output feeds into the next.
    Useful for complex workflows like:
    - Research -> Outline -> Draft -> Review
    - Analyze -> Plan -> Execute -> Verify
    - Translate -> Localize -> Format
    """

    def __init__(self, name: str, providers: dict):
        self.name = name
        self.providers = providers
        self.steps: list[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self

    async def execute(self, initial_input: str, context: dict = None) -> PipelineResult:
        """Execute the full pipeline with the given input."""
        context = context or {}
        current_input = initial_input
        step_results = []
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0.0

        for i, step in enumerate(self.steps):
            # Check condition
            if step.condition and not step.condition(current_input, context):
                logger.info(f"Skipping step '{step.name}' (condition not met)")
                continue

            logger.info(f"Pipeline '{self.name}' step {i+1}/{len(self.steps)}: {step.name}")

            provider = self.providers.get(step.provider)
            if not provider:
                return PipelineResult(
                    steps=step_results,
                    final_output=current_input,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    total_latency_ms=total_latency,
                    success=False,
                    error=f"Provider '{step.provider}' not found",
                )

            messages = [{"role": "user", "content": current_input}]
            if step.system_prompt:
                messages.insert(0, {"role": "system", "content": step.system_prompt})

            request = CompletionRequest(
                messages=messages,
                model=step.model,
                max_tokens=step.max_tokens,
                temperature=step.temperature,
            )

            try:
                start = time.monotonic()
                response = await provider.complete(request)
                latency = (time.monotonic() - start) * 1000

                output = response.content
                if step.transform:
                    output = step.transform(output)

                step_results.append({
                    "step": step.name,
                    "provider": step.provider,
                    "model": response.model,
                    "input_preview": current_input[:100],
                    "output_preview": output[:100],
                    "tokens": response.total_tokens,
                    "cost": response.cost_usd,
                    "latency_ms": latency,
                })

                total_tokens += response.total_tokens
                total_cost += response.cost_usd
                total_latency += latency
                current_input = output

            except Exception as e:
                logger.error(f"Pipeline step '{step.name}' failed: {e}")
                return PipelineResult(
                    steps=step_results,
                    final_output=current_input,
                    total_tokens=total_tokens,
                    total_cost_usd=total_cost,
                    total_latency_ms=total_latency,
                    success=False,
                    error=f"Step '{step.name}' failed: {e}",
                )

        return PipelineResult(
            steps=step_results,
            final_output=current_input,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 6),
            total_latency_ms=round(total_latency, 1),
            success=True,
        )


# Pre-built pipeline templates
def create_research_pipeline(providers: dict) -> Pipeline:
    """Research -> Analyze -> Summarize pipeline."""
    pipe = Pipeline("research", providers)
    pipe.add_step(PipelineStep(
        name="research",
        provider="openai",
        model="gpt-4o",
        system_prompt="You are a research assistant. Find key facts and data points.",
        max_tokens=2048,
    ))
    pipe.add_step(PipelineStep(
        name="analyze",
        provider="xiaomi",
        model="mimo-v2.5-pro",
        system_prompt="You are an analyst. Analyze the research and identify patterns, insights, and implications.",
        max_tokens=2048,
    ))
    pipe.add_step(PipelineStep(
        name="summarize",
        provider="anthropic",
        model="claude-sonnet-4-0",
        system_prompt="You are a technical writer. Create a clear, structured summary with key takeaways.",
        max_tokens=1024,
    ))
    return pipe


def create_code_review_pipeline(providers: dict) -> Pipeline:
    """Analyze -> Review -> Suggest pipeline for code review."""
    pipe = Pipeline("code_review", providers)
    pipe.add_step(PipelineStep(
        name="analyze_code",
        provider="anthropic",
        model="claude-sonnet-4-0",
        system_prompt="Analyze the code for structure, patterns, and potential issues.",
        max_tokens=2048,
    ))
    pipe.add_step(PipelineStep(
        name="security_review",
        provider="xiaomi",
        model="mimo-v2.5-pro",
        system_prompt="Review for security vulnerabilities, injection risks, and unsafe patterns.",
        max_tokens=1024,
    ))
    pipe.add_step(PipelineStep(
        name="suggest_fixes",
        provider="anthropic",
        model="claude-sonnet-4-0",
        system_prompt="Provide specific, actionable code improvement suggestions with examples.",
        max_tokens=2048,
    ))
    return pipe
