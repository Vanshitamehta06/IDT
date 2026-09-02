"""Side-by-side multi-paper comparative analysis — two-phase sequential pipeline.

Phase 1: For each paper in turn, run a short extraction prompt (~2 500 chars).
         The model answers 10 structured questions about one paper at a time.
         Runs SEQUENTIALLY — Ollama is single-threaded; concurrent calls just queue.

Phase 2: Feed the pre-extracted fact sheets into a compact comparison prompt.
         The model only has to write a table from two small fact lists, not parse
         thousands of chars of raw evidence[cite: 31].
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple, Optional

from llm.llm_model import OllamaLLM
from llm.prompt_builder import bound_text, comparison_prompt

_EXTRACT_EVIDENCE_MAX = 3000   # chars of evidence fed per extraction call
_EXTRACT_MAX_PROMPT   = 4500   # total chars for extraction prompt
_COMPARE_MAX_PROMPT   = 6000   # total chars for comparison prompt[cite: 31]
_MAX_PAPERS_TO_COMPARE = 4     # Cap to prevent context window overload in Phase 2

logger = logging.getLogger(__name__)


def _short_label(filename: str, index: int) -> str:
    """Return a short human-readable label like 'Paper A (rag_survey.pdf)'[cite: 31]."""
    letter = chr(ord("A") + index)
    base = filename.split("\\")[-1].split("/")[-1]
    display = base if len(base) <= 40 else base[:37] + "…"
    return f"Paper {letter} ({display})"


def _extraction_prompt(short_label: str, evidence: str) -> str:
    ev = bound_text(evidence, _EXTRACT_EVIDENCE_MAX)
    return f"""Read the evidence from ONE research paper and answer these 10 questions.
Each answer must be ONE sentence taken from the evidence only.
If the answer is not in the evidence, write: Not found

Paper: {short_label}

1. Research problem:
2. Objective:
3. Method/Approach:
4. Model/Architecture:
5. Datasets or benchmarks:
6. Evaluation metric(s):
7. Key results (numbers if reported):
8. Main contributions:
9. Limitations stated by the authors:
10. Future work mentioned:

Evidence:
{ev}
"""


def _compare_prompt(paper_facts: List[Tuple[str, str]]) -> str:
    """Build the Phase-2 comparison prompt from extracted fact sheets[cite: 31]."""
    col_headers = " | ".join(lbl for lbl, _ in paper_facts)
    # FIX: Corrected Markdown table separator spacing
    sep = " | ".join(["---"] * (len(paper_facts) + 1))
    
    # FIX: Corrected row placeholder spacing
    placeholders = " | ".join(["?"] * len(paper_facts))

    fact_blocks = ""
    for label, facts in paper_facts:
        fact_blocks += f"\n### {label} — extracted facts\n{facts}\n"

    return f"""Using ONLY the extracted facts below, write a side-by-side comparison.

{fact_blocks}

---
Write the following:

### Comparison table

| Aspect | {col_headers} |
| {sep} |
| Research problem | {placeholders} |
| Objective | {placeholders} |
| Method / Approach | {placeholders} |
| Model / Architecture | {placeholders} |
| Datasets / Benchmarks | {placeholders} |
| Evaluation metrics | {placeholders} |
| Key results | {placeholders} |
| Contributions | {placeholders} |
| Limitations | {placeholders} |
| Future work | {placeholders} |

Replace every ? with the matching fact from the correct paper column.
Write "Not found" when the extracted fact says "Not found".
NEVER put Paper A facts in Paper B columns or vice versa.

### Key similarities

### Key differences in approach

### Results compared
Name which paper reports better results and on which metric/dataset.

### Strengths and weaknesses of each approach
Name each paper explicitly.

### Research gap revealed by this comparison
"""


class ComparisonService:
    def __init__(self, llm: OllamaLLM) -> None:
        self.llm = llm

    async def compare(
        self,
        query: str,
        evidence: str,
        per_paper_evidence: Optional[Dict[str, str]] = None,
    ) -> str:
        if per_paper_evidence and len(per_paper_evidence) >= 2:
            # Enforce paper limit to prevent prompt overflow
            if len(per_paper_evidence) > _MAX_PAPERS_TO_COMPARE:
                per_paper_evidence = dict(list(per_paper_evidence.items())[:_MAX_PAPERS_TO_COMPARE])
            return await self._two_phase_compare(per_paper_evidence)
            
        # Fallback: single-block (legacy / no paper_filter)[cite: 31]
        return await self.llm.agenerate(
            comparison_prompt(query, evidence),
            system="Compare papers using only the evidence. Name each paper explicitly.",
            max_prompt_chars=_COMPARE_MAX_PROMPT,
        )

    async def _two_phase_compare(
        self, per_paper_evidence: Dict[str, str]
    ) -> str:
        labels_raw = list(per_paper_evidence.keys())
        short_labels = [_short_label(fname, i) for i, fname in enumerate(labels_raw)]

        # ── Phase 1: extract facts for each paper SEQUENTIALLY ─────────────
        # Ollama is single-threaded — concurrent calls just queue and timeout[cite: 31].
        extractions: List[Tuple[str, str]] = []
        for i, fname in enumerate(labels_raw):
            lbl = short_labels[i]
            evidence = per_paper_evidence[fname]
            prompt = _extraction_prompt(lbl, evidence)
            
            # FIX: Added robust error handling to prevent single paper failure from halting comparison
            try:
                raw = await self.llm.agenerate(
                    prompt,
                    system=(
                        "Answer each numbered question in one sentence from the evidence only. "
                        "Write 'Not found' when the evidence does not contain the answer. "
                        "Never guess or invent."
                    ),
                    max_prompt_chars=_EXTRACT_MAX_PROMPT,
                )
                extractions.append((lbl, raw))
            except Exception as e:
                logger.error(f"Failed to extract facts for {lbl}: {e}")
                extractions.append((lbl, "Extraction failed. Not found."))

        # ── Phase 2: compare the extracted fact sheets ──────────────────────
        compare_text = _compare_prompt(extractions)
        try:
            comparison = await self.llm.agenerate(
                compare_text,
                system=(
                    "Fill the comparison table from the extracted facts. "
                    "Use Paper A / Paper B labels. Never mix facts between papers. "
                    "Write 'Not found' for missing items."
                ),
                max_prompt_chars=_COMPARE_MAX_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to generate comparison table: {e}")
            comparison = "**Error:** Failed to generate final comparison due to model timeout or context constraints."

        # Prepend individual fact sheets so the user can verify the extraction[cite: 31]
        parts = ["## Paper-by-paper extraction\n"]
        for lbl, facts in extractions:
            parts.append(f"### {lbl}\n\n{facts}\n")
        parts.append("---\n\n## Comparison\n\n" + comparison)
        
        return "\n".join(parts)