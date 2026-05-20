"""Tests for the pipeline chain."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.pipeline.chain import Pipeline, PipelineStep, PipelineResult
from agentflow.providers.base import CompletionResponse


@pytest.fixture
def mock_providers():
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=CompletionResponse(
        content="step output",
        model="test-model",
        provider="test",
        total_tokens=100,
        cost_usd=0.001,
        latency_ms=200,
    ))
    return {"test": provider}


class TestPipeline:
    @pytest.mark.asyncio
    async def test_single_step(self, mock_providers):
        pipe = Pipeline("test", mock_providers)
        pipe.add_step(PipelineStep(
            name="step1",
            provider="test",
            model="test-model",
        ))
        result = await pipe.execute("input text")
        assert result.success
        assert result.final_output == "step output"
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_multi_step(self, mock_providers):
        pipe = Pipeline("test", mock_providers)
        pipe.add_step(PipelineStep(name="s1", provider="test", model="m1"))
        pipe.add_step(PipelineStep(name="s2", provider="test", model="m2"))
        pipe.add_step(PipelineStep(name="s3", provider="test", model="m3"))
        result = await pipe.execute("input")
        assert result.success
        assert len(result.steps) == 3
        assert result.total_tokens == 300

    @pytest.mark.asyncio
    async def test_transform(self, mock_providers):
        pipe = Pipeline("test", mock_providers)
        pipe.add_step(PipelineStep(
            name="transform_step",
            provider="test",
            model="test-model",
            transform=lambda x: x.upper(),
        ))
        result = await pipe.execute("hello")
        assert result.success

    @pytest.mark.asyncio
    async def test_missing_provider(self, mock_providers):
        pipe = Pipeline("test", mock_providers)
        pipe.add_step(PipelineStep(name="bad", provider="nonexistent", model="x"))
        result = await pipe.execute("input")
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_skip_condition(self, mock_providers):
        pipe = Pipeline("test", mock_providers)
        pipe.add_step(PipelineStep(
            name="conditional",
            provider="test",
            model="test-model",
            condition=lambda inp, ctx: False,
        ))
        result = await pipe.execute("input")
        assert result.success
        assert len(result.steps) == 0
