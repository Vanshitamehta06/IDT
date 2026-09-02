"""Research gap and underexplored-area identifier."""

from __future__ import annotations

from llm.llm_model import OllamaLLM
from llm.prompt_builder import gap_prompt


class GapAnalysisService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def find_gaps(self, query: str, evidence: str) -> str:
        return await self.llm.agenerate(
            gap_prompt(query, evidence),
            system=(
                "Identify research gaps from the uploaded paper only. "
                "Skip any chunk describing software code, Streamlit, or application files. "
                "Every gap must be supported by a specific citation from the evidence. "
                "Never invent gaps. Use 'the paper' (singular)."
            ),
        )
