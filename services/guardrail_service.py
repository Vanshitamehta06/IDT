"""Input and output guardrails for the research assistant.

Key change from previous version
---------------------------------
Realtime queries (weather, prices, live scores) are NO LONGER hard-blocked.
They are flagged with blocked_reason="realtime_info_required" but passed=True
so the orchestrator can route them to a helpful realtime-fallback response
rather than returning a blank answer.

Only genuinely off-topic queries (recipes, sports gossip, etc.) and empty
queries are hard-blocked.

Output guardrails
-----------------
- Grounding check: is the answer actually supported by retrieved context?
- Unsupported-claim detection: flags the response when grounding is low.
"""

from __future__ import annotations

import re
from typing import Any

from api.schemas import GuardrailResult
from config import GUARDRAIL_SCOPE_KEYWORDS, GUARDRAIL_SCOPE_THRESHOLD

_WORD = re.compile(r"[a-z0-9_]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_REFUSAL_PHRASES = (
    "insufficient",
    "do not have",
    "not in the evidence",
    "cannot find",
    "not found",
    "no information",
    "not available",
    "not provided",
)

# Realtime queries: NOT hard-blocked — flagged so orchestrator can handle gracefully
_REALTIME_RE = re.compile(
    r"\b(weather|today['s]?|right now|current price|stock price|forex|bitcoin price|"
    r"breaking news|latest news|live score|what time is it|current time|"
    r"exchange rate|cryptocurrency|crypto price)\b",
    re.I,
)

# Genuinely off-topic: hard-blocked (these have zero research angle)
_OFF_TOPIC_HARD = re.compile(
    r"\b(recipe for|how to cook|how to bake|celebrity gossip|horoscope for|"
    r"lottery numbers|movie review of|song lyrics for|write me a poem about|"
    r"tell me a joke)\b",
    re.I,
)


def _tokenize(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _scope_score(query: str) -> float:
    """0–1 fraction of scope keywords that appear in the query."""
    if not GUARDRAIL_SCOPE_KEYWORDS:
        return 1.0
    tokens = _tokenize(query)
    hits = sum(1 for kw in GUARDRAIL_SCOPE_KEYWORDS if kw.lower() in tokens or kw.lower() in query.lower())
    return round(hits / len(GUARDRAIL_SCOPE_KEYWORDS), 4)


def check_input(query: str, indexed_chunks: int = 0) -> GuardrailResult:
    """Validate the incoming query before sending it into the pipeline.

    Returns GuardrailResult with:
      passed=True   → proceed normally
      passed=False  → hard block (only for truly empty / off-topic-hard)
      blocked_reason="realtime_info_required" + passed=True → soft flag,
        orchestrator should produce a helpful "I can't provide live data" response.
    """
    q = (query or "").strip()

    # 1. Empty
    if not q:
        return GuardrailResult(
            passed=False,
            reason="Query is empty.",
            blocked_reason="out_of_scope",
        )

    # 2. Hard off-topic (no research angle whatsoever)
    if _OFF_TOPIC_HARD.search(q):
        return GuardrailResult(
            passed=False,
            reason=(
                "This Research Assistant is focused on academic papers, retrieval-augmented "
                "generation, NLP, and codebase understanding. "
                "That question is outside its scope."
            ),
            scope_score=0.0,
            blocked_reason="out_of_scope",
        )

    # 3. Realtime query — SOFT FLAG, not a hard block
    #    The orchestrator will handle it with a helpful explanation.
    if _REALTIME_RE.search(q):
        score = _scope_score(q)
        return GuardrailResult(
            passed=True,                       # allow through — don't blank-answer
            reason="Real-time information required.",
            scope_score=score,
            blocked_reason="realtime_info_required",  # signal to orchestrator
        )

    score = _scope_score(q)

    # 4. No sources indexed — soft warning, not a hard block
    if indexed_chunks == 0:
        return GuardrailResult(
            passed=True,
            reason="No documents indexed yet. Upload and index PDFs for grounded answers.",
            scope_score=score,
        )

    return GuardrailResult(passed=True, scope_score=score)


def check_output(
    answer: str,
    retrieved: list[dict[str, Any]],
    grounding_threshold: float = 0.35,
) -> tuple[float, bool]:
    """Return (grounding_score 0–1, flagged bool).

    grounding_score: fraction of answer sentences that are supported by retrieved context.
    flagged: True when grounding_score < grounding_threshold.
    """
    if not (answer or "").strip():
        return 0.0, True

    lowered = answer.lower()
    if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
        return 1.0, False

    context = " ".join(str(c.get("text") or "") for c in (retrieved or [])).lower()
    if not context:
        return 0.0, len(answer.split()) > 20

    ctx_tokens = _tokenize(context)
    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()]
    if not sentences:
        sentences = [answer]

    grounded = 0
    for sent in sentences:
        overlap = _tokenize(sent) & ctx_tokens
        content = {t for t in overlap if len(t) > 2}   # lowered from >3
        if len(content) >= 2:                           # lowered from >=4
            grounded += 1

    score = round(grounded / max(len(sentences), 1), 4)
    return score, score < grounding_threshold
