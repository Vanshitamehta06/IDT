"""Metric definitions and scoring for the three-model evaluation.

How each metric is calculated
-----------------------------
Correctness / Accuracy
    Mean over items of: fraction of `gold_keywords` that appear in the answer
    (case-insensitive). If `must_cite` and no [Source N] / [Page tag], score
    is multiplied by 0.7. Range 0–1.

Relevance
    Token Jaccard overlap between the question and the answer. Range 0–1.

Retrieval Quality
    For items with `expected_files`, 1 if any retrieved chunk filename contains
    an expected path fragment; else keyword hit-rate of gold_keywords inside
    concatenated retrieved chunk text. Range 0–1. Items with no retrieval
    target still score keyword coverage of retrieved text.

Hallucination Rate
    1 minus groundedness. A sentence is grounded if at least 4 content tokens
    overlap the retrieved context (or the answer explicitly refuses). Rate is
    ungrounded_sentences / max(sentences, 1). Lower is better.

Test-Pass Rate
    Only for items with `code_test`. Fraction of those items whose extracted
    Python passes the deterministic unit check. Not applicable items are
    excluded from the denominator.

Response Latency
    Wall-clock milliseconds for the full orchestrator.run call (`elapsed_ms`).

Token Usage
    prompt_eval_count + eval_count from Ollama when available; otherwise
    estimated as len(raw_prompt+answer)/4.

CPU / GPU / Memory
    Process RSS (MB) and CPU percent from psutil after the call; GPU via
    nvidia-smi when present. Reported as mean RSS and mean CPU across items.
"""

from __future__ import annotations

import re
from typing import Any

_WORD = re.compile(r"[a-z0-9_./:-]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[source\s+\d+\]|\[page\s+\d+", re.I)


def tokenize(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    blob = (text or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in blob)
    return hits / len(keywords)


def correctness(answer: str, item: dict[str, Any]) -> float:
    score = keyword_hit_rate(answer, item.get("gold_keywords") or [])
    if item.get("must_cite") and not _CITE.search(answer or ""):
        score *= 0.7
    return round(score, 4)


def relevance(question: str, answer: str) -> float:
    q, a = tokenize(question), tokenize(answer)
    if not q or not a:
        return 0.0
    return round(len(q & a) / len(q | a), 4)


def retrieval_quality(item: dict[str, Any], retrieved: list[dict[str, Any]]) -> float:
    files = [str(c.get("filename") or "") for c in retrieved]
    expected = item.get("expected_files") or []
    if expected:
        hits = 0
        for exp in expected:
            if any(exp.replace("\\", "/") in f.replace("\\", "/") for f in files):
                hits += 1
        return round(hits / len(expected), 4)
    blob = " ".join(str(c.get("text") or "") for c in retrieved)
    return round(keyword_hit_rate(blob, item.get("gold_keywords") or []), 4)


def hallucination_rate(answer: str, retrieved: list[dict[str, Any]]) -> float:
    context = " ".join(str(c.get("text") or "") for c in retrieved).lower()
    if not (answer or "").strip():
        return 1.0
    lowered = answer.lower()
    if any(p in lowered for p in ("insufficient", "do not have", "not in the evidence", "cannot find")):
        return 0.0
    sentences = [s.strip() for s in _SENTENCE.split(answer) if s.strip()]
    if not sentences:
        sentences = [answer]
    ungrounded = 0
    ctx_tokens = tokenize(context)
    for sent in sentences:
        overlap = tokenize(sent) & ctx_tokens
        content = {t for t in overlap if len(t) > 3}
        if len(content) < 4 and not _CITE.search(sent):
            ungrounded += 1
    return round(ungrounded / max(len(sentences), 1), 4)


def classify_retrieval(item: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    rq = retrieval_quality(item, retrieved)
    if not retrieved:
        return "important_information_missed"
    if rq >= 0.67:
        return "relevant_information_retrieved"
    if rq <= 0.15:
        return "irrelevant_information_retrieved"
    return "partial_or_mixed_retrieval"


def score_item(item: dict[str, Any], answer: str, retrieved: list[dict[str, Any]], code_result: dict[str, Any]) -> dict[str, Any]:
    corr = correctness(answer, item)
    hall = hallucination_rate(answer, retrieved)
    return {
        "correctness": corr,
        "relevance": relevance(item.get("question") or "", answer),
        "retrieval_quality": retrieval_quality(item, retrieved),
        "hallucination_rate": hall,
        "code_test": code_result,
        "retrieval_label": classify_retrieval(item, retrieved),
        "answer_correct_enough": corr >= 0.5,
        "hallucinated_despite_context": hall >= 0.5 and retrieval_quality(item, retrieved) >= 0.5,
    }
