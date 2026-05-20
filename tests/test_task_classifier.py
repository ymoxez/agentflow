"""Tests for the task classifier."""

import pytest
from agentflow.router.task_classifier import classify_task, TaskCategory


class TestTaskClassifier:
    def test_code_classification(self):
        result = classify_task("Write a Python function to sort a list")
        assert result.category == TaskCategory.CODE
        assert result.confidence > 0

    def test_reasoning_classification(self):
        result = classify_task("Analyze the trade-offs between REST and GraphQL")
        assert result.category == TaskCategory.REASONING

    def test_math_classification(self):
        result = classify_task("Solve the equation 3x + 5 = 20")
        assert result.category == TaskCategory.MATH

    def test_creative_classification(self):
        result = classify_task("Write a creative short story about a robot discovering emotions. Include vivid imagery and dialogue.")
        assert result.category == TaskCategory.CREATIVE

    def test_analysis_classification(self):
        result = classify_task("Summarize this dataset and extract key metrics")
        assert result.category == TaskCategory.ANALYSIS

    def test_translation_classification(self):
        result = classify_task("Translate this text to Japanese")
        assert result.category == TaskCategory.TRANSLATION

    def test_chat_fallback(self):
        result = classify_task("Hello, how are you?")
        assert result.category == TaskCategory.CHAT

    def test_has_suggested_model(self):
        result = classify_task("Write a Python function")
        assert result.suggested_model
        assert result.suggested_provider

    def test_code_with_debug(self):
        result = classify_task("Debug this error: TypeError in line 42")
        assert result.category == TaskCategory.CODE

    def test_mixed_uses_highest_score(self):
        result = classify_task(
            "Write a Python function that analyzes data and explains the algorithm"
        )
        assert result.category in (TaskCategory.CODE, TaskCategory.ANALYSIS)
