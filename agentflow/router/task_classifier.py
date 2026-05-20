"""
Task classifier — automatically categorizes prompts to route to optimal model.

Categories: code, reasoning, creative, analysis, chat, translation, math
Uses lightweight classification to avoid wasting tokens on routing.
"""

import re
from enum import Enum
from dataclasses import dataclass


class TaskCategory(Enum):
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    CHAT = "chat"
    TRANSLATION = "translation"
    MATH = "math"
    VISION = "vision"


@dataclass
class ClassificationResult:
    category: TaskCategory
    confidence: float
    suggested_model: str
    suggested_provider: str
    reasoning: str


# Keyword patterns for fast classification
_PATTERNS = {
    TaskCategory.CODE: [
        r"\b(write|fix|debug|refactor|implement|code|function|class|api|endpoint)\b",
        r"\b(python|javascript|typescript|rust|go|java|sql|html|css)\b",
        r"\b(bug|error|exception|traceback|stack trace)\b",
        r"```",
    ],
    TaskCategory.REASONING: [
        r"\b(analyze|explain why|reason|think step|chain of thought|consider)\b",
        r"\b(pros and cons|trade-?offs?|compare|evaluate|assess)\b",
        r"\b(strategy|plan|approach|design decision)\b",
    ],
    TaskCategory.CREATIVE: [
        r"\b(write a story|poem|creative|imagine|fiction|narrative|dialogue)\b",
        r"\b(blog post|article|copywriting|marketing|headline)\b",
        r"\b(brainstorm|ideate|inspire)\b",
    ],
    TaskCategory.MATH: [
        r"\b(calculate|compute|solve|equation|formula|proof|theorem)\b",
        r"\b(integral|derivative|matrix|probability|statistics)\b",
        r"[0-9]+[\+\-\*\/\^][0-9]+",
    ],
    TaskCategory.TRANSLATION: [
        r"\b(translate|translation|convert to|in [a-z]+ language)\b",
        r"\b(meaning of .+ in .+)\b",
    ],
    TaskCategory.ANALYSIS: [
        r"\b(summarize|summary|extract|parse|structure|organize)\b",
        r"\b(data|dataset|csv|json|report|metrics)\b",
        r"\b(review|audit|inspect)\b",
    ],
    TaskCategory.VISION: [
        r"\b(describe this image|what is in|analyze this photo|screenshot)\b",
        r"\b(ocr|read text from|identify)\b",
    ],
}


# Model routing table — maps category to best model
_MODEL_ROUTING = {
    TaskCategory.CODE: ("claude-sonnet-4-0", "anthropic"),
    TaskCategory.REASONING: ("mimo-v2.5-pro", "xiaomi"),
    TaskCategory.CREATIVE: ("gpt-4o", "openai"),
    TaskCategory.ANALYSIS: ("mimo-v2.5-pro", "xiaomi"),
    TaskCategory.CHAT: ("mimo-v2.5-lite", "xiaomi"),
    TaskCategory.TRANSLATION: ("deepseek-chat", "deepseek"),
    TaskCategory.MATH: ("deepseek-reasoner", "deepseek"),
    TaskCategory.VISION: ("gpt-4o", "openai"),
}


def classify_task(prompt: str) -> ClassificationResult:
    """
    Classify a user prompt into a task category using keyword patterns.
    Returns the category, confidence score, and suggested model.
    """
    prompt_lower = prompt.lower()
    scores: dict[TaskCategory, float] = {}

    for category, patterns in _PATTERNS.items():
        score = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            score += len(matches) * 0.3
        scores[category] = min(score, 1.0)

    # Pick highest scoring category
    if not any(s > 0 for s in scores.values()):
        best = TaskCategory.CHAT
        confidence = 0.5
    else:
        best = max(scores, key=scores.get)
        confidence = scores[best]

    model, provider = _MODEL_ROUTING[best]

    return ClassificationResult(
        category=best,
        confidence=confidence,
        suggested_model=model,
        suggested_provider=provider,
        reasoning=f"Matched {best.value} patterns with confidence {confidence:.2f}",
    )
