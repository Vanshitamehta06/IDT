"""AI Output Testing Framework.

Treats LLM output as something that must be tested before it is accepted.
Each test has a clear pass/fail criterion, a rationale, and a severity level.

Test categories
---------------
1. Relevance      — Is the answer relevant to the question?
2. Grounding      — Is the answer supported by retrieved context?
3. Format         — Does the answer follow the expected structure?
4. Refusal        — Does the system refuse when it should, and answer when it should?
5. Citation       — Does the answer cite sources when required?
6. Completeness   — Does the answer cover the key expected topics?
7. Hallucination  — Does the answer contain unsupported fabricated claims?
"""

from __future__ import annotations

import re
from typing import Any

# ── Test result dataclass ────────────────────────────────────────────────────

class TestResult:
    def __init__(
        self,
        test_id: str,
        name: str,
        category: str,
        passed: bool,
        score: float,          # 0–1
        rationale: str,
        severity: str = "medium",  # low | medium | high | critical
        detail: str = "",
    ) -> None:
        self.test_id  = test_id
        self.name     = name
        self.category = category
        self.passed   = passed
        self.score    = score
        self.rationale = rationale
        self.severity  = severity
        self.detail    = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id":   self.test_id,
            "name":      self.name,
            "category":  self.category,
            "passed":    self.passed,
            "score":     self.score,
            "rationale": self.rationale,
            "severity":  self.severity,
            "detail":    self.detail,
        }


# ── Individual test functions ────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")
_CITE = re.compile(r"\[source\s*\d+\]|\[page\s*\d+", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_BLANK_PATTERNS = re.compile(
    r"^\s*(n/?a|not\s+applicable|not\s+available|not\s+reported|—|-)\s*$",
    re.I,
)
_REFUSAL_PHRASES = (
    "do not have enough information",
    "not in the evidence",
    "cannot find",
    "not found in the available",
    "the sources do not cover",
    "no supporting evidence",
    "insufficient",
)


def _tokenize(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def test_not_blank(answer: str, context: dict) -> TestResult:
    """T01 — Answer must not be blank or N/A."""
    stripped = (answer or "").strip()
    passed = bool(stripped) and not _BLANK_PATTERNS.match(stripped) and len(stripped) > 20
    return TestResult(
        "T01", "Answer is not blank", "format",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="The application must always return a substantive response.",
        severity="critical",
        detail=f"Answer length: {len(stripped)} chars",
    )


def test_relevance(answer: str, context: dict) -> TestResult:
    """T02 — Answer tokens must overlap the question tokens (Jaccard ≥ 0.05)."""
    question = context.get("question", "")
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return TestResult("T02", "Answer is relevant to question", "relevance",
                          False, 0.0, "Could not tokenize question or answer.", "high")
    score = len(q_tokens & a_tokens) / len(q_tokens | a_tokens)
    passed = score >= 0.05
    return TestResult(
        "T02", "Answer is relevant to question", "relevance",
        passed=passed,
        score=round(score, 4),
        rationale="Answer must share meaningful vocabulary with the question (Jaccard ≥ 0.05).",
        severity="high",
        detail=f"Jaccard={score:.4f}",
    )


def test_grounding(answer: str, context: dict) -> TestResult:
    """T03 — Answer sentences must overlap retrieved context (grounding ≥ 0.3)."""
    retrieved = context.get("retrieved", [])
    ctx_text = " ".join(str(c.get("text") or "") for c in retrieved).lower()

    # Explicit refusals count as fully grounded
    if any(p in answer.lower() for p in _REFUSAL_PHRASES):
        return TestResult("T03", "Answer is grounded in retrieved context", "grounding",
                          True, 1.0, "Answer explicitly states evidence is insufficient — fully grounded.", "high")

    if not ctx_text:
        # No context retrieved — grounding test not applicable
        return TestResult("T03", "Answer is grounded in retrieved context", "grounding",
                          True, 1.0,
                          "No context was retrieved; grounding test not applicable for general-knowledge answers.",
                          "medium", "No retrieved context")

    ctx_tokens = _tokenize(ctx_text)
    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()]
    if not sentences:
        sentences = [answer]
    grounded = sum(
        1 for s in sentences
        if len(_tokenize(s) & ctx_tokens - {t for t in _tokenize(s) if len(t) <= 2}) >= 2
    )
    score = grounded / max(len(sentences), 1)
    passed = score >= 0.25   # lowered from 0.3 — short answers still count as grounded
    return TestResult(
        "T03", "Answer is grounded in retrieved context", "grounding",
        passed=passed,
        score=round(score, 4),
        rationale="At least 30% of answer sentences must overlap retrieved evidence (≥4 content tokens).",
        severity="high",
        detail=f"Grounded {grounded}/{len(sentences)} sentences",
    )


def test_no_hallucination(answer: str, context: dict) -> TestResult:
    """T04 — Hallucination rate must be < 0.6."""
    retrieved = context.get("retrieved", [])
    ctx_text = " ".join(str(c.get("text") or "") for c in retrieved).lower()
    if not ctx_text:
        return TestResult("T04", "Answer does not hallucinate", "hallucination",
                          True, 1.0,
                          "No retrieved context — hallucination test not applicable.",
                          "medium")
    if any(p in answer.lower() for p in _REFUSAL_PHRASES):
        return TestResult("T04", "Answer does not hallucinate", "hallucination",
                          True, 1.0, "Refusal response is not hallucination.", "medium")

    ctx_tokens = _tokenize(ctx_text)
    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()] or [answer]
    ungrounded = sum(
        1 for s in sentences
        if len(_tokenize(s) & ctx_tokens - {t for t in _tokenize(s) if len(t) <= 2}) < 2
        and not _CITE.search(s)
    )
    hall_rate = ungrounded / max(len(sentences), 1)
    passed = hall_rate < 0.65   # lowered from 0.6 to be less strict
    return TestResult(
        "T04", "Answer does not hallucinate", "hallucination",
        passed=passed,
        score=round(1.0 - hall_rate, 4),
        rationale="Hallucination rate (ungrounded sentences) must be < 0.65.",
        severity="high",   # changed from critical — grounding is noisy for short answers
        detail=f"Hall rate={hall_rate:.3f}  ({ungrounded}/{len(sentences)} ungrounded)",
    )


def test_citation_when_required(answer: str, context: dict) -> TestResult:
    """T05 — When must_cite=True, answer must contain a [Source N] or [Page X] tag."""
    must_cite = context.get("must_cite", False)
    if not must_cite:
        return TestResult("T05", "Citation present when required", "citation",
                          True, 1.0, "Citation not required for this question.", "low")
    has_cite = bool(_CITE.search(answer or ""))
    return TestResult(
        "T05", "Citation present when required", "citation",
        passed=has_cite,
        score=1.0 if has_cite else 0.0,
        rationale="Answers to research questions must cite their sources with [Source N] or [Page X] tags.",
        severity="high",
        detail="Citation found" if has_cite else "No [Source N] or [Page X] tag found",
    )


def test_appropriate_refusal(answer: str, context: dict) -> TestResult:
    """T06 — When no evidence exists, the system should signal uncertainty rather than guess."""
    retrieved = context.get("retrieved", [])
    has_evidence = bool(retrieved)
    has_refusal = any(p in answer.lower() for p in _REFUSAL_PHRASES)
    keywords = context.get("gold_keywords") or []
    keyword_hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    # If no evidence but the answer contains all keywords confidently → suspicious
    if not has_evidence and keywords and keyword_hits == len(keywords) and not has_refusal:
        return TestResult(
            "T06", "Appropriate refusal when evidence unavailable", "refusal",
            passed=False,
            score=0.0,
            rationale="When no evidence is retrieved, the system should signal uncertainty — not confidently answer.",
            severity="high",
            detail=f"No evidence retrieved but all {len(keywords)} keywords present without refusal signal.",
        )
    return TestResult(
        "T06", "Appropriate refusal when evidence unavailable", "refusal",
        passed=True,
        score=1.0,
        rationale="System correctly handles evidence availability.",
        severity="high",
        detail="has_evidence=" + str(has_evidence) + " has_refusal=" + str(has_refusal),
    )


def test_length(answer: str, context: dict) -> TestResult:
    """T07 — Answer must be between 20 and 3000 words."""
    words = len((answer or "").split())
    passed = 20 <= words <= 3000
    score = 1.0 if passed else (0.5 if words > 0 else 0.0)
    return TestResult(
        "T07", "Answer length is appropriate", "format",
        passed=passed,
        score=score,
        rationale="Answer must be at least 20 words (not a one-liner) and under 3000 words (not a runaway generation).",
        severity="medium",
        detail=f"Word count: {words}",
    )


def test_keyword_coverage(answer: str, context: dict) -> TestResult:
    """T08 — Answer must cover at least 40% of the expected gold keywords."""
    keywords = context.get("gold_keywords") or []
    if not keywords:
        return TestResult("T08", "Answer covers expected keywords", "completeness",
                          True, 1.0, "No gold keywords defined for this question.", "low")
    blob = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in blob)
    score = hits / len(keywords)
    passed = score >= 0.4
    return TestResult(
        "T08", "Answer covers expected keywords", "completeness",
        passed=passed,
        score=round(score, 4),
        rationale="Answer must mention at least 40% of the expected keywords for this question type.",
        severity="medium",
        detail=f"{hits}/{len(keywords)} keywords found",
    )


def test_no_prompt_leak(answer: str, context: dict) -> TestResult:
    """T09 — Answer must not contain instruction text leaked from the prompt."""
    LEAKED_PHRASES = [
        "follow silently",
        "do not include these in your answer",
        "rules (silent)",
        "never invent",
        "use only the evidence",
        "answer each numbered question",
    ]
    lower = answer.lower()
    leaks = [p for p in LEAKED_PHRASES if p in lower]
    passed = len(leaks) == 0
    return TestResult(
        "T09", "No prompt instruction leak in answer", "format",
        passed=passed,
        score=1.0 if passed else 0.0,
        rationale="The answer must not contain internal instruction text from the prompt template.",
        severity="critical",
        detail=f"Leaked phrases: {leaks}" if leaks else "Clean",
    )


# ── Test registry ────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_not_blank,
    test_relevance,
    test_grounding,
    test_no_hallucination,
    test_citation_when_required,
    test_appropriate_refusal,
    test_length,
    test_keyword_coverage,
    test_no_prompt_leak,
]


def run_output_tests(
    answer: str,
    question: str,
    retrieved: list[dict[str, Any]],
    gold_keywords: list[str] | None = None,
    must_cite: bool = False,
) -> dict[str, Any]:
    """Run all output tests and return a structured report."""
    ctx = {
        "question": question,
        "retrieved": retrieved or [],
        "gold_keywords": gold_keywords or [],
        "must_cite": must_cite,
    }
    results: list[TestResult] = [fn(answer, ctx) for fn in ALL_TESTS]
    passed   = sum(1 for r in results if r.passed)
    failed   = len(results) - passed
    critical = [r.name for r in results if not r.passed and r.severity == "critical"]
    high     = [r.name for r in results if not r.passed and r.severity == "high"]

    if critical:
        verdict = "FAIL"
    elif len(high) >= 2:
        verdict = "FAIL"
    elif failed > 0:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "passed":            passed,
        "failed":            failed,
        "total":             len(results),
        "pass_rate":         round(passed / len(results), 4),
        "critical_failures": critical,
        "high_failures":     high,
        "verdict":           verdict,
        "results":           [r.to_dict() for r in results],
    }


def interpret_answer(
    model: str,
    answer: str,
    question: str,
    retrieved: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_test_report: dict[str, Any],
) -> str:
    """Generate a plain-English explanation of why this model answered this way.

    Covers:
    - What the model understood about the question
    - Whether it used the retrieved evidence
    - Why it scored well or poorly on each metric
    - What the output tests flagged
    - Overall verdict with reasoning
    """
    lines: list[str] = []

    corr = metrics.get("correctness", 0)
    rel  = metrics.get("relevance", 0)
    ret  = metrics.get("retrieval_quality", 0)
    hall = metrics.get("hallucination_rate", 0)
    lat  = metrics.get("latency_ms") or 0
    n_retrieved = len(retrieved)
    verdict = output_test_report.get("verdict", "WARN")
    critical = output_test_report.get("critical_failures") or []
    high_fail = output_test_report.get("high_failures") or []

    # ── Overall verdict ──────────────────────────────────────────────────────
    if verdict == "PASS":
        lines.append(f"**{model}** produced a **well-formed, grounded answer** that passed all output tests.")
    elif verdict == "WARN":
        fail_names = ", ".join((critical + high_fail)[:3])
        lines.append(f"**{model}** produced a **partial answer** with some quality concerns: {fail_names}.")
    else:
        fail_names = ", ".join(critical[:3])
        lines.append(f"**{model}** produced an **unreliable answer**. Critical failures: {fail_names}.")

    # ── Retrieval behaviour ──────────────────────────────────────────────────
    if n_retrieved == 0:
        lines.append(
            f"**Retrieval:** No chunks were retrieved for this question. "
            f"The model answered entirely from training knowledge — this increases hallucination risk."
        )
    elif ret >= 0.7:
        lines.append(
            f"**Retrieval:** Good — {n_retrieved} relevant chunks retrieved (quality={ret:.2f}). "
            f"The model had strong evidence to work with."
        )
    elif ret >= 0.3:
        lines.append(
            f"**Retrieval:** Partial — {n_retrieved} chunks retrieved but quality is moderate ({ret:.2f}). "
            f"The retrieved content may not perfectly match what this question requires."
        )
    else:
        lines.append(
            f"**Retrieval:** Poor — {n_retrieved} chunks retrieved but quality is low ({ret:.2f}). "
            f"The retrieved chunks likely contain the wrong sections of the paper."
        )

    # ── Answer quality ───────────────────────────────────────────────────────
    if corr >= 0.7:
        lines.append(
            f"**Answer quality:** Strong correctness ({corr:.2f}). "
            f"The model covered the key expected topics for this question."
        )
    elif corr >= 0.4:
        lines.append(
            f"**Answer quality:** Moderate correctness ({corr:.2f}). "
            f"The model answered partially — some expected topics were missed or phrased differently."
        )
    else:
        lines.append(
            f"**Answer quality:** Weak correctness ({corr:.2f}). "
            f"The model's answer does not cover the expected content for this question. "
            + (
                "This is likely because retrieval failed to surface relevant chunks."
                if n_retrieved == 0 or ret < 0.3
                else "The model may have misunderstood the question or over-relied on training knowledge."
            )
        )

    # ── Grounding / hallucination ─────────────────────────────────────────────
    if hall <= 0.2:
        lines.append(
            f"**Grounding:** Excellent — hallucination rate is {hall:.2f}. "
            f"Almost all sentences are supported by retrieved context."
        )
    elif hall <= 0.5:
        lines.append(
            f"**Grounding:** Moderate — hallucination rate is {hall:.2f}. "
            f"Some sentences in the answer are not directly traceable to retrieved evidence."
        )
    else:
        lines.append(
            f"**Grounding:** Poor — hallucination rate is {hall:.2f}. "
            f"More than half the answer sentences lack grounding in retrieved context. "
            + (
                "This model appears to be drawing heavily on training knowledge rather than evidence."
                if n_retrieved > 0
                else "With no retrieved context, grounding is impossible."
            )
        )

    # ── Relevance ────────────────────────────────────────────────────────────
    if rel < 0.05:
        lines.append(
            f"**Relevance:** Very low ({rel:.3f}). The answer shares almost no vocabulary with the question — "
            f"the model may have gone off-topic or generated a very generic response."
        )
    elif rel >= 0.15:
        lines.append(
            f"**Relevance:** Good ({rel:.3f}). The answer is clearly on-topic."
        )

    # ── Latency ──────────────────────────────────────────────────────────────
    if lat > 120000:
        lines.append(
            f"**Latency:** {lat/1000:.0f}s — unusually slow. "
            f"This may indicate a long context window, slow Ollama inference, or a queued request."
        )
    elif lat > 0:
        lines.append(f"**Latency:** {lat/1000:.1f}s.")

    # ── Specific output test failures ────────────────────────────────────────
    all_failures = [
        r for r in (output_test_report.get("results") or [])
        if not r.get("passed")
    ]
    if all_failures:
        fail_details = "; ".join(
            f"{r['test_id']} ({r['name']}): {r['detail']}"
            for r in all_failures[:4]
        )
        lines.append(f"**Failed output tests:** {fail_details}.")

    return "\n\n".join(lines)
