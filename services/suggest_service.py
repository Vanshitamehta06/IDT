"""Dynamic prompt suggestion service.

Generates 4–5 contextual research question suggestions based on:
1. The partial query the user has typed
2. The titles of indexed PDFs (domain context)
3. A curated template library grouped by research topic

The suggestions are purely heuristic (no LLM call) to keep latency < 50 ms.
They are ranked by token overlap with the partial query.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Template library — grouped by trigger keywords
# ---------------------------------------------------------------------------
_TEMPLATES: list[tuple[tuple[str, ...], list[str]]] = [
    # RAG / retrieval
    (
        ("retrieval", "rag", "retrieve", "search", "vector", "embed"),
        [
            "How does retrieval-augmented generation improve factual accuracy?",
            "What are the main limitations of dense retrieval in RAG systems?",
            "Compare sparse BM25 and dense vector retrieval for academic question answering.",
            "How should retrieved context be ranked and filtered before generation?",
            "What retrieval quality metrics are used to evaluate RAG pipelines?",
        ],
    ),
    # Hallucination
    (
        ("hallucin", "grounding", "unsupported", "fabricat", "invent"),
        [
            "How can hallucination in language model outputs be detected automatically?",
            "What evidence grounding techniques reduce hallucination in RAG?",
            "Compare hallucination rates across different model sizes on the same knowledge base.",
            "What does a hallucinated answer look like versus a grounded answer?",
            "How does retrieval quality affect hallucination rate?",
        ],
    ),
    # Language models / LLMs
    (
        ("llm", "language model", "gpt", "llama", "transformer", "model"),
        [
            "What are the key differences between parametric and non-parametric knowledge in LLMs?",
            "How do small local LLMs compare to large hosted models for research QA?",
            "What prompt engineering techniques improve factual accuracy in local LLMs?",
            "Summarize the architecture and training approach described in the local knowledge base.",
            "Which model performs best for code generation on the benchmark in the indexed papers?",
        ],
    ),
    # Evaluation / benchmarks
    (
        ("evaluat", "benchmark", "metric", "measure", "score", "accuracy", "performance"),
        [
            "What evaluation metrics are used for RAG systems according to the local papers?",
            "How is retrieval quality measured separately from generation quality?",
            "Compare correctness and relevance scores across the three evaluation models.",
            "What benchmarks does the literature use for academic question answering?",
            "How should a multi-model RAG evaluation be designed to be fair?",
        ],
    ),
    # Gaps / future work
    (
        ("gap", "open problem", "future", "limitation", "unsolved", "underexplore"),
        [
            "What open research gaps remain in multi-document academic RAG?",
            "What limitations do the indexed papers identify in their own methods?",
            "Which aspects of retrieval-augmented generation are underexplored according to the literature?",
            "What future work directions do the authors propose?",
            "What problems remain unsolved in citation-grounded question answering?",
        ],
    ),
    # Comparison
    (
        ("compare", "versus", "vs", "differ", "contrast", "better", "advantage"),
        [
            "Compare the methodologies used in the papers in the local collection.",
            "Compare dense retrieval versus sparse BM25 for academic RAG systems.",
            "What are the trade-offs between model size and answer quality in this pipeline?",
            "Compare the hallucination rates of the three evaluation models.",
            "How do the papers in the library differ in their approach to knowledge grounding?",
        ],
    ),
    # Summary / analysis
    (
        ("summar", "analyz", "overview", "explain", "break down", "describe"),
        [
            "Summarize the methodology and key findings from the indexed PDFs.",
            "Analyze the core problem studied across the local knowledge base.",
            "Give a literature landscape of the topics covered in the indexed papers.",
            "Explain how the RAG pipeline in this application is structured end-to-end.",
            "What are the main contributions and limitations of the indexed research?",
        ],
    ),
    # Code / implementation
    (
        ("code", "implement", "function", "class", "module", "service", "file", "repo", "codebase"),
        [
            "Which files are involved in handling a research query from HTTP request to final answer?",
            "How does the hybrid re-ranking combine dense search and BM25 in this codebase?",
            "What would be affected if the citation service were modified?",
            "Explain the chunking strategy used in the ingestion pipeline.",
            "Which components handle intent classification and query routing?",
        ],
    ),
    # Citations / sources
    (
        ("cite", "citation", "source", "reference", "evidence", "support"),
        [
            "How should answers be cited according to the indexed knowledge base?",
            "What citation format does this application use for retrieved evidence?",
            "Show the supporting evidence for the claim that dense retrieval outperforms BM25.",
            "Find direct quotes from the indexed PDFs about evaluation methodology.",
            "Which sources support the argument for hybrid retrieval strategies?",
        ],
    ),
    # Recommendations
    (
        ("recommend", "read next", "reading list", "what should", "suggest paper"),
        [
            "Recommend the next three papers to read on citation-grounded question answering.",
            "What related work should I explore after reading the indexed RAG survey?",
            "Suggest papers that address the limitations identified in the local collection.",
            "What papers are most relevant to improving retrieval quality in academic RAG?",
            "Give a ranked reading list for someone new to retrieval-augmented generation.",
        ],
    ),
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _rank_suggestion(suggestion: str, partial_tokens: set[str], partial: str) -> float:
    s_tokens = _tokens(suggestion)
    jaccard = len(partial_tokens & s_tokens) / max(len(partial_tokens | s_tokens), 1)
    seq = _similarity(partial, suggestion[:60])
    return jaccard * 0.7 + seq * 0.3


def _domain_suggestions_from_titles(titles: list[str], partial: str) -> list[str]:
    """Generate suggestions that reference the actual indexed paper titles."""
    if not titles:
        return []
    pt = _tokens(partial)
    results: list[str] = []
    for title in titles[:4]:
        short = title.replace(".pdf", "").replace("_", " ").title()
        candidates = [
            f"Summarize the methodology and findings in {short}.",
            f"What are the key contributions of {short}?",
            f"What limitations does {short} identify in its approach?",
            f"Compare the method in {short} with other approaches in the collection.",
        ]
        for c in candidates:
            score = _rank_suggestion(c, pt, partial)
            if score > 0.02:
                results.append((score, c))
    results.sort(key=lambda x: -x[0])
    return [r[1] for r in results[:3]]


def suggest(
    partial: str,
    context_titles: list[str] | None = None,
    n: int = 5,
) -> list[str]:
    """Return up to *n* ranked question suggestions for the partial query."""
    partial = (partial or "").strip()
    if not partial:
        return []

    partial_tokens = _tokens(partial)
    scored: list[tuple[float, str]] = []

    # Score all template suggestions
    for triggers, suggestions in _TEMPLATES:
        # Boost if any trigger keyword matches the partial
        trigger_boost = any(t in partial.lower() for t in triggers)
        for suggestion in suggestions:
            score = _rank_suggestion(suggestion, partial_tokens, partial)
            if trigger_boost:
                score = min(score + 0.25, 1.0)
            scored.append((score, suggestion))

    # Add domain-specific suggestions from indexed PDF titles
    for ds in _domain_suggestions_from_titles(context_titles or [], partial):
        scored.append((0.45, ds))  # give domain suggestions moderate baseline priority

    # Sort by score descending, deduplicate
    scored.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    result: list[str] = []
    for _, suggestion in scored:
        if suggestion not in seen and len(result) < n:
            seen.add(suggestion)
            result.append(suggestion)

    return result
