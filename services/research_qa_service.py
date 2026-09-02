"""Grounded Q&A over retrieved local (and registered) evidence."""

from __future__ import annotations

from llm.llm_model import OllamaLLM
from llm.prompt_builder import format_history, qa_prompt


class ResearchQAService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def answer(self, query: str, evidence: str, history: list[dict[str, str]] | None = None) -> str:
        prompt = qa_prompt(query, evidence, format_history(history))
        return await self.llm.agenerate(
            prompt,
            system="Ground every claim in the evidence. Prefer [Source N] and [Page X | Section Y].",
        )
