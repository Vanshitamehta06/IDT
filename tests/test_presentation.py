from services.presentation_service import infer_kind, parse_markdown_table, split_sections
from api.schemas import IntentType


def test_infer_comparison() -> None:
    assert infer_kind(IntentType.RESEARCH_QA, "Compare RAG vs fine-tuning") == "comparison"


def test_parse_table() -> None:
    text = """
| A | B |
| --- | --- |
| 1 | 2 |
"""
    headers, rows = parse_markdown_table(text)
    assert headers == ["A", "B"]
    assert rows == [["1", "2"]]


def test_split_quick_take() -> None:
    sections = split_sections("### Quick Take\nHello world.\n### Why it matters\nBecause.")
    assert "quick take" in sections
    assert "hello" in sections["quick take"].lower()
