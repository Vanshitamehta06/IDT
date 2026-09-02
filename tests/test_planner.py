"""Lightweight tests for planner JSON extraction and context bounding."""

from llm.prompt_builder import bound_text
from services.planner_service import extract_json_object, heuristic_plan
from api.schemas import IntentType


def test_extract_json_from_fences() -> None:
    raw = """Sure. Here you go:
```json
{"intent": "discovery", "tools": ["arxiv_search"], "arxiv_query": "RAG survey", "requires_local_rag": false, "comparison_targets": [], "rationale": "find papers"}
```
"""
    parsed = extract_json_object(raw)
    assert parsed is not None
    assert parsed["intent"] == "discovery"


def test_extract_json_nested_and_noise() -> None:
    raw = 'Preface {"intent": "comparison", "tools": ["rag_search", "comparison"], "arxiv_query": "", "requires_local_rag": true, "comparison_targets": ["A"], "rationale": "x"} thanks'
    parsed = extract_json_object(raw)
    assert parsed["intent"] == "comparison"


def test_heuristic_gap() -> None:
    plan = heuristic_plan("What are the open research gaps in RAG?")
    assert plan.intent == IntentType.GAP_ANALYSIS


def test_bound_text_truncates() -> None:
    text = bound_text("abcdefghij" * 10, 12)
    assert len(text) <= 12
    assert text.endswith("…")
