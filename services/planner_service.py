"""Intent classification and execution planning with robust JSON extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from api.schemas import ExecutionPlan, IntentType
from llm.llm_model import OllamaLLM
from config import PLANNER_TIMEOUT_SECONDS, USE_LLM_PLANNER
from llm.prompt_builder import bound_text, format_history, planner_prompt

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

INTENT_KEYWORDS: dict[IntentType, tuple[str, ...]] = {
    # ANALYSIS must come before GAP_ANALYSIS because summarize/methodology
    # queries should route to analysis, not gap detection.
    IntentType.ANALYSIS: (
        "summarize",
        "summarise",
        "summary of",
        "summarization",
        "analyze",
        "analyse",
        "methodology",
        "contributions",
        "break down",
        "structure",
        "key findings",
        "main findings",
        "what does the paper",
        "what does this paper",
        "overview of the paper",
        "paper analysis",
        "paper summary",
    ),
    IntentType.COMPARISON: (
        "compare",
        "versus",
        "vs",
        "difference",
        "side by side",
        "which is better",
        "advantages",
        "disadvantages",
        "trade-off",
        "tradeoff",
    ),
    IntentType.GAP_ANALYSIS: ("gap", "open problem", "underexplored", "future work", "limitation", "missing", "unresolved", "research direction", "what is missing"),
    IntentType.RECOMMENDATION: ("recommend", "what should i read", "reading list", "next paper"),
    IntentType.DISCOVERY: (
        "arxiv",
        "find papers",
        "search literature",
        "recent papers",
        "discover",
        "landscape",
        "literature map",
    ),
}

TOOL_MAP: dict[IntentType, list[str]] = {
    IntentType.RESEARCH_QA: ["rag_search", "research_qa"],
    IntentType.DISCOVERY: ["arxiv_search", "rag_search", "research_qa"],
    IntentType.ANALYSIS: ["rag_search", "analysis"],
    IntentType.COMPARISON: ["rag_search", "arxiv_search", "comparison"],
    IntentType.GAP_ANALYSIS: ["rag_search", "arxiv_search", "gap_analysis"],
    IntentType.RECOMMENDATION: ["arxiv_search", "rag_search", "recommendation"],
}


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    fenced = _FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    candidates = [cleaned]
    match = _JSON_RE.search(cleaned)
    if match:
        candidates.append(match.group(0))
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def heuristic_plan(query: str) -> ExecutionPlan:
    lowered = query.lower()
    intent = IntentType.RESEARCH_QA
    for candidate, keywords in INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            intent = candidate
            break
    return ExecutionPlan(
        intent=intent,
        tools=list(TOOL_MAP[intent]),
        arxiv_query=query if intent in {IntentType.DISCOVERY, IntentType.RECOMMENDATION, IntentType.COMPARISON, IntentType.GAP_ANALYSIS} else "",
        requires_local_rag=True,
        rationale="Heuristic fallback after JSON parse failure.",
    )


class PlannerService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def plan(self, query: str, history: list[dict[str, str]] | None = None) -> ExecutionPlan:
        fallback = heuristic_plan(query)
        if not USE_LLM_PLANNER:
            fallback.rationale = "Fast heuristic planner (set USE_LLM_PLANNER=true to use Ollama)."
            return fallback
        prompt = planner_prompt(bound_text(query, 1500), format_history(history))
        try:
            raw = await asyncio.wait_for(
                self.llm.agenerate(
                    prompt,
                    system="Return only compact JSON. No markdown, no commentary.",
                    temperature=0.0,
                ),
                timeout=PLANNER_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("Planner LLM timed out after %ss; using heuristic", PLANNER_TIMEOUT_SECONDS)
            return fallback
        parsed = extract_json_object(raw)
        if not parsed:
            logger.warning("Planner JSON parse failed; using heuristic")
            return fallback
        return self._coerce(parsed, query)

    def _coerce(self, payload: dict[str, Any], query: str) -> ExecutionPlan:
        intent_raw = str(payload.get("intent") or "research_qa").strip().lower()
        try:
            intent = IntentType(intent_raw)
        except ValueError:
            intent = heuristic_plan(query).intent

        tools = payload.get("tools") or TOOL_MAP[intent]
        if isinstance(tools, str):
            tools = [tools]
        tools = [str(t).strip() for t in tools if str(t).strip()]
        if not tools:
            tools = list(TOOL_MAP[intent])

        targets = payload.get("comparison_targets") or []
        if isinstance(targets, str):
            targets = [targets]

        requires_rag = payload.get("requires_local_rag", True)
        if isinstance(requires_rag, str):
            requires_rag = requires_rag.strip().lower() in {"1", "true", "yes"}

        return ExecutionPlan(
            intent=intent,
            tools=tools,
            arxiv_query=str(payload.get("arxiv_query") or "").strip() or (
                query if "arxiv_search" in tools else ""
            ),
            requires_local_rag=bool(requires_rag),
            comparison_targets=[str(t) for t in targets],
            rationale=str(payload.get("rationale") or "")[:400],
        )
