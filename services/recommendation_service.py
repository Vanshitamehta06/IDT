"""Literature recommendation engine over local + discovery candidates."""

from __future__ import annotations

from llm.llm_model import OllamaLLM
from llm.prompt_builder import recommendation_prompt


class RecommendationService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def recommend(self, query: str, evidence: str) -> str:
        return await self.llm.agenerate(
            recommendation_prompt(query, evidence),
            system=(
                "Recommend academic papers or research directions based on the uploaded paper. "
                "Do NOT recommend Python source files, Streamlit code, or application configs as papers. "
                "Every recommendation must connect to the paper's actual research problem or methods. "
                "Label general suggestions clearly. Never invent paper titles or authors."
            ),
        )
