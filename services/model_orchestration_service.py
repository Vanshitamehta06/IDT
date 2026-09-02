"""Intelligent model orchestration — complexity classification and model selection.

Decision flow
-------------
1. classify_complexity(query) → ComplexityLevel (SIMPLE | COMPLEX)
2. select_model(complexity, available_models, preferred) → model tag string
3. The orchestrator uses this to (optionally) route to a different OllamaLLM instance.
4. After generation, check_grounding() validates the answer against retrieved context.

Complexity heuristics (no extra LLM call — fast, deterministic)
----------------------------------------------------------------
- COMPLEX if query has >= COMPLEXITY_TOKEN_THRESHOLD tokens
- COMPLEX if any COMPLEXITY_KEYWORDS are found in the query
- COMPLEX for intents: comparison, gap_analysis, analysis
- SIMPLE otherwise

Model-tier mapping
------------------
The service maintains a priority list derived from the configured EVAL_MODELS.
- COMPLEX → first model in the tier list (assumed most capable)
- SIMPLE  → last model (assumed fastest / lightest)
If only one model is available, it is used for both tiers.
"""

from __future__ import annotations

import logging

from api.schemas import ComplexityLevel, ModelOrchestrationDecision
from config import (
    COMPLEXITY_KEYWORDS,
    COMPLEXITY_TOKEN_THRESHOLD,
    EVAL_MODELS,
    GROUNDING_THRESHOLD,
    MODEL_TIER_COMPLEX,
    MODEL_TIER_SIMPLE,
    OLLAMA_MODEL,
)
from services.guardrail_service import check_output

logger = logging.getLogger(__name__)

# Intents that warrant the more capable model tier
_COMPLEX_INTENTS = {"comparison", "gap_analysis", "analysis"}


def classify_complexity(query: str, intent: str = "research_qa") -> ComplexityLevel:
    """Classify a query as SIMPLE or COMPLEX using fast heuristics."""
    tokens = query.split()
    if len(tokens) >= COMPLEXITY_TOKEN_THRESHOLD:
        return ComplexityLevel.COMPLEX
    lowered = query.lower()
    if any(kw in lowered for kw in COMPLEXITY_KEYWORDS):
        return ComplexityLevel.COMPLEX
    if intent in _COMPLEX_INTENTS:
        return ComplexityLevel.COMPLEX
    return ComplexityLevel.SIMPLE


def select_model(
    complexity: ComplexityLevel,
    available_models: list[str] | None = None,
    preferred_model: str | None = None,
) -> tuple[str, str]:
    """Return (model_tag, rationale).

    Priority order:
    1. preferred_model if supplied (user chose explicitly)
    2. Tier-specific env vars (MODEL_TIER_SIMPLE / MODEL_TIER_COMPLEX)
    3. EVAL_MODELS ordering (first = most capable, last = lightest)
    4. OLLAMA_MODEL fallback
    """
    if preferred_model:
        return preferred_model, "User-selected model."

    candidates = available_models or EVAL_MODELS or [OLLAMA_MODEL]

    # Env-var overrides
    if complexity == ComplexityLevel.SIMPLE and MODEL_TIER_SIMPLE:
        model = MODEL_TIER_SIMPLE
        rationale = f"Simple query → lightweight model ({model}) from MODEL_TIER_SIMPLE."
        return model, rationale
    if complexity == ComplexityLevel.COMPLEX and MODEL_TIER_COMPLEX:
        model = MODEL_TIER_COMPLEX
        rationale = f"Complex query → capable model ({model}) from MODEL_TIER_COMPLEX."
        return model, rationale

    if not candidates:
        return OLLAMA_MODEL, "No candidates; using default OLLAMA_MODEL."

    if complexity == ComplexityLevel.SIMPLE:
        # Use the last (lightest) available model
        model = candidates[-1]
        rationale = (
            f"Simple query ({len(model.split())} tokens) → "
            f"fastest model ({model}). Prioritising low latency."
        )
    else:
        # Use the first (most capable) available model
        model = candidates[0]
        rationale = (
            f"Complex query → most capable available model ({model}). "
            "Prioritising accuracy over speed."
        )
    return model, rationale


def build_orchestration_decision(
    query: str,
    intent: str,
    selected_model: str,
    rationale: str,
    answer: str | None = None,
    retrieved: list[dict] | None = None,
) -> ModelOrchestrationDecision:
    """Build the full orchestration decision including grounding validation."""
    complexity = classify_complexity(query, intent)

    grounding_score = 1.0
    grounding_flagged = False
    if answer and retrieved is not None:
        grounding_score, grounding_flagged = check_output(
            answer, retrieved, grounding_threshold=GROUNDING_THRESHOLD
        )

    return ModelOrchestrationDecision(
        complexity=complexity,
        selected_model=selected_model,
        rationale=rationale,
        grounding_score=grounding_score,
        grounding_flagged=grounding_flagged,
    )
