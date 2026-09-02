"""Unified evidence registry with deduplicated [Source N] tags."""

from __future__ import annotations

from api.schemas import ArxivPaper, Citation, RetrievedChunk
from llm.prompt_builder import bound_text


class CitationService:
    def __init__(self) -> None:
        self._items: list[Citation] = []
        self._keys: dict[str, int] = {}

    def reset(self) -> None:
        self._items = []
        self._keys = {}

    def register_chunk(self, chunk: RetrievedChunk) -> Citation:
        locator_bits = []
        if chunk.page is not None:
            locator_bits.append(f"Page {chunk.page}")
        if chunk.section:
            locator_bits.append(f"Section {chunk.section}")
        locator = " | ".join(locator_bits)
        key = f"pdf::{chunk.filename}::{chunk.page}::{chunk.section}::{chunk.text[:80]}"
        return self._upsert(
            key=key,
            kind="local_pdf",
            title=chunk.filename or "Local PDF",
            locator=locator,
            # Store up to 600 chars so abstract chunks show actual content,
            # not just author lists or section headers.
            snippet=bound_text(chunk.text, 700),
            filename=chunk.filename,
            page=chunk.page,
            section=chunk.section,
        )

    def register_paper(self, paper: ArxivPaper) -> Citation:
        key = f"arxiv::{paper.arxiv_id or paper.title}"
        return self._upsert(
            key=key,
            kind="arxiv",
            title=paper.title,
            locator=paper.arxiv_id,
            snippet=bound_text(paper.abstract, 280),
            url=paper.url,
            arxiv_id=paper.arxiv_id,
        )

    def _upsert(self, *, key: str, **fields) -> Citation:
        if key in self._keys:
            return self._items[self._keys[key] - 1]
        source_id = len(self._items) + 1
        citation = Citation(
            source_id=source_id,
            tag=f"[Source {source_id}]",
            **fields,
        )
        self._keys[key] = source_id
        self._items.append(citation)
        return citation

    def as_list(self) -> list[Citation]:
        return list(self._items)

    def evidence_block(self, max_chars: int = 3500) -> str:
        lines: list[str] = []
        used = 0
        # Sort: local_pdf items with lower page numbers first (abstract/intro
        # before results/experiments), then arxiv, preserving original order
        # within each group.  This ensures definitional content (abstract) reaches
        # the LLM before results-section noise even when retrieval ranks them lower.
        def _sort_key(cite: Citation) -> tuple:
            kind_order = 0 if cite.kind == "local_pdf" else 1
            page = cite.page or 999
            return (kind_order, page)

        sorted_items = sorted(self._items, key=_sort_key)

        for cite in sorted_items:
            parts: list[str] = []
            if cite.page is not None:
                parts.append(f"Page {cite.page}")
            if cite.section:
                parts.append(f"Section {cite.section}")
            locator = f" [{' | '.join(parts)}]" if parts else ""
            # For comparison, use up to 600 chars per snippet to include
            # methodology and results sections fully.
            snippet = bound_text(cite.snippet, 600)
            block = f"{cite.tag} {cite.title}{locator}\n{snippet}".strip()
            if used + len(block) + 2 > max_chars:
                break
            lines.append(block)
            used += len(block) + 2
        return "\n\n".join(lines)
