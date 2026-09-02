"""Structural paper analysis: problem, method, contributions, limitations."""

from __future__ import annotations

from llm.llm_model import OllamaLLM
from llm.prompt_builder import analysis_prompt


class AnalysisService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def analyze(self, query: str, evidence: str) -> str:
        return await self.llm.agenerate(
            analysis_prompt(query, evidence),
            system=(
                "You are summarizing an academic research paper. "
                "Use ONLY the paper content in the evidence. "
                "Ignore any chunks that describe software code, Streamlit UI, FastAPI, "
                "evaluation scripts, or application configuration — those are not part of the paper. "
                "Use Markdown headings. Cite [Source N] and [Page X | Section Y]. "
                "Never invent results or paper metadata."
            ),
        )
