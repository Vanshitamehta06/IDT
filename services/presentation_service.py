"""Turn raw LLM answers + citations into intent-aware presentation payloads."""

from __future__ import annotations

import re
from typing import Any

from api.schemas import ArxivPaper, Citation, IntentType, RetrievedChunk

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
NOT_REPORTED = "Not reported"


def infer_kind(intent: IntentType | str, query: str) -> str:
    q = (query or "").lower()
    intent_value = intent.value if isinstance(intent, IntentType) else str(intent)
    compare_bits = (
        "compare",
        " versus ",
        " vs ",
        " vs.",
        "difference between",
        "which is better",
        "advantages",
        "disadvantages",
        "trade-off",
        "tradeoff",
    )
    if any(bit in q for bit in compare_bits) or intent_value == "comparison":
        if "paper" in q or "study" in q or "article" in q:
            return "paper_comparison"
        return "comparison"
    if any(bit in q for bit in ("gap", "open problem", "underexplored", "future work")) or intent_value == "gap_analysis":
        return "gaps"
    if any(bit in q for bit in ("recommend", "what should i read", "reading list", "next paper")) or intent_value == "recommendation":
        return "recommendations"
    if any(bit in q for bit in ("overview", "landscape", "literature map", "survey of")):
        return "overview"
    if any(bit in q for bit in ("summar", "methodology", "limitations", "contributions")) or intent_value == "analysis":
        return "analysis"
    return "factual"


def task_label(kind: str) -> str:
    return {
        "comparison": "Comparison",
        "paper_comparison": "Paper comparison",
        "gaps": "Research gaps",
        "recommendations": "Recommendations",
        "overview": "Literature overview",
        "analysis": "Paper analysis",
        "factual": "Evidence review",
    }.get(kind, "Evidence review")


def tabs_for(kind: str) -> list[str]:
    if kind in {"comparison", "paper_comparison"}:
        return ["Overview", "Comparison", "Evidence", "Sources"]
    if kind == "gaps":
        return ["Overview", "Research Gaps", "Evidence", "Sources"]
    if kind == "recommendations":
        return ["Overview", "Recommendations", "Evidence", "Sources"]
    if kind == "overview":
        return ["Overview", "Landscape", "Evidence", "Sources"]
    if kind == "analysis":
        return ["Overview", "Analysis", "Evidence", "Sources"]
    return ["Answer", "Evidence", "Sources"]


def parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]]:
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and "---" not in stripped:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells:
                rows.append(cells)
    if len(rows) < 2:
        return [], []
    headers = rows[0]
    body = [r for r in rows[1:] if len(r) == len(headers)]
    return headers, body


def split_sections(text: str) -> dict[str, str]:
    if not text:
        return {}
    matches = list(_HEADING.finditer(text))
    if not matches:
        return {"body": text.strip()}
    sections: dict[str, str] = {}
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["_preamble"] = preamble
    for i, match in enumerate(matches):
        title = match.group(2).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[title] = text[start:end].strip()
    return sections


def first_paragraphs(text: str, n: int = 3) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if parts:
        return "\n\n".join(parts[:2])
    sentences = [s.strip() for s in _SENTENCE.split(text or "") if s.strip()]
    return " ".join(sentences[:n])


def _section(sections: dict[str, str], *names: str) -> str:
    for name in names:
        for key, value in sections.items():
            if name in key:
                return value
    return ""


def _fill_missing(cell: str) -> str:
    value = (cell or "").strip()
    if not value or value.lower() in {"n/a", "na", "-", "unknown", "none"}:
        return NOT_REPORTED
    return value


def build_presentation(
    *,
    query: str,
    answer: str,
    intent: IntentType,
    citations: list[Citation],
    papers: list[ArxivPaper],
    chunks: list[RetrievedChunk] | None = None,
    indexed_docs: int = 0,
) -> dict[str, Any]:
    kind = infer_kind(intent, query)
    sections = split_sections(answer)
    headers, table_rows = parse_markdown_table(answer)
    table_rows = [[_fill_missing(c) for c in row] for row in table_rows]

    quick = _section(sections, "quick take", "executive summary", "key insight", "answer")
    if not quick:
        # Use preamble or first paragraphs, but strip any markdown heading lines
        raw = sections.get("_preamble") or ""
        if not raw:
            raw = (
                _section(sections, "answer")
                or _section(sections, "what the paper says about this")
                or _section(sections, "what the papers say")
                or first_paragraphs(answer)
            )
        import re as _re
        raw_clean = _re.sub(r"^\s*#{1,3}\s+.*\n?", "", raw, flags=_re.MULTILINE).strip()
        quick = first_paragraphs(raw_clean) if raw_clean else first_paragraphs(answer)

    # For analysis/summary, do NOT populate findings/why from QA-style headings.
    # The Analysis tab renders sections directly, and duplicating into findings
    # would cause the Overview tab to show the same content twice.
    if kind == "analysis":
        findings = ""
        why = _section(sections, "conclusion / significance", "significance", "conclusion", "why it matters")
    else:
        why = _section(sections, "why it matters", "what this means", "significance", "general context")
        # "findings" feeds the "What the papers say" block in render_overview.
        # The new prompts use "### Answer" + "### Supporting evidence".
        # Map "answer" → quick_take (done above) and "supporting evidence" → findings
        # so it renders in the Evidence tab, not duplicated in Overview.
        # Never populate findings from large explanation blocks (>400 chars).
        _excl = _section(sections, "what the paper says about this")
        findings = _section(sections, "supporting evidence", "what the papers say", "key findings", "core problem")
        if findings and _excl and findings.strip() == _excl.strip():
            findings = ""
        if findings and len(findings) > 400:
            findings = ""

    comparison = None
    if kind in {"comparison", "paper_comparison"}:
        if headers and table_rows:
            comparison = {
                "headers": headers,
                "rows": table_rows,
                "bottom_line": _section(sections, "bottom line", "synthesis", "short synthesis") or first_paragraphs(answer, 2),
            }
        else:
            names = []
            for paper in papers[:4]:
                names.append(paper.title[:48] or paper.arxiv_id)
            if not names:
                seen = []
                for cite in citations:
                    label = cite.filename or cite.title
                    if label and label not in seen:
                        seen.append(label)
                names = seen[:3] or ["Source A", "Source B"]
            default_headers = ["Criteria", *names]
            criteria = [
                "Core approach",
                "Accuracy",
                "Computational cost",
                "Data requirement",
                "Strengths",
                "Limitations",
                "Best suited for",
            ]
            comparison = {
                "headers": default_headers,
                "rows": [[c, *[NOT_REPORTED] * (len(default_headers) - 1)] for c in criteria],
                "bottom_line": _section(sections, "bottom line")
                or "The sources do not support a complete numeric comparison. Treat unfilled cells as Not reported.",
                "incomplete": True,
            }

    paper_rows = []
    if kind == "paper_comparison":
        paper_headers = ["Paper", "Year", "Problem", "Method", "Dataset", "Key Finding", "Limitation"]
        if headers and table_rows and len(headers) >= 5:
            paper_headers, paper_body = headers, table_rows
        else:
            paper_body = []
            for paper in papers[:6]:
                year = (paper.published or "")[:4] or NOT_REPORTED
                paper_body.append(
                    [
                        paper.title or NOT_REPORTED,
                        year,
                        NOT_REPORTED,
                        NOT_REPORTED,
                        NOT_REPORTED,
                        (paper.abstract or NOT_REPORTED)[:180],
                        NOT_REPORTED,
                    ]
                )
            if not paper_body:
                for cite in citations[:6]:
                    paper_body.append(
                        [
                            cite.title or cite.filename or NOT_REPORTED,
                            NOT_REPORTED,
                            NOT_REPORTED,
                            NOT_REPORTED,
                            NOT_REPORTED,
                            cite.snippet or NOT_REPORTED,
                            NOT_REPORTED,
                        ]
                    )
            paper_headers = ["Paper", "Year", "Problem", "Method", "Dataset", "Key Finding", "Limitation"]
        paper_rows = {"headers": paper_headers, "rows": paper_body}

    gaps = []
    if kind == "gaps":
        # The gap_prompt produces structured blocks like:
        #   ### Gap 01
        #   **Limitation:** ...
        #   **Evidence:** ...
        #   **Why it matters:** ...
        #   **Possible direction:** ...
        # Split on Gap headings then extract each field.
        import re as _gre
        gap_blocks = _gre.split(
            r"(?:^|\n)#{1,3}\s*Gap\s*0*(\d+)\s*\n",
            answer, flags=_gre.IGNORECASE
        )
        # gap_blocks alternates: [preamble, "1", block1_text, "2", block2_text, ...]
        idx = 1
        i = 0
        # Skip preamble (index 0), then process pairs (number, text)
        items = gap_blocks[1:]  # drop preamble
        while len(items) >= 2 and idx <= 6:
            _block_text = items[1].strip()
            items = items[2:]

            def _field(text: str, *labels: str) -> str:
                for lbl in labels:
                    m = _gre.search(
                        r"\*\*" + _gre.escape(lbl) + r"[:：]\*\*\s*(.*?)(?=\n\*\*|\Z)",
                        text, _gre.DOTALL | _gre.IGNORECASE
                    )
                    if m:
                        return m.group(1).strip()
                return ""

            limitation = _field(_block_text, "Limitation", "Gap", "Issue")
            evidence   = _field(_block_text, "Evidence")
            why        = _field(_block_text, "Why it matters", "Why")
            direction  = _field(_block_text, "Possible direction", "Direction", "Future work")

            # Fallback: if no labelled fields, use first paragraph as limitation
            if not limitation:
                limitation = first_paragraphs(_block_text, 1)

            if len(limitation) >= 20:
                gaps.append({
                    "id": f"{idx:02d}",
                    "limitation": limitation,
                    "evidence":   evidence,
                    "why":        why,
                    "direction":  direction,
                })
                idx += 1

        # Legacy fallback: older format without labelled fields
        if not gaps:
            gap_text = _section(sections, "gap 01", "gap 1", "research gap", "open research", "gaps") or answer
            legacy_blocks = re.split(r"(?:^|\n)(?:#{1,3}\s*)?(?:gap\s*)?0?\d+[.)]\s*", gap_text, flags=re.I)
            idx = 1
            for block in legacy_blocks:
                clean = block.strip()
                if len(clean) < 40:
                    continue
                gaps.append({
                    "id": f"{idx:02d}",
                    "limitation": first_paragraphs(clean, 2),
                    "evidence": "",
                    "why": "",
                    "direction": "",
                })
                idx += 1
                if idx > 6:
                    break

        if not gaps:
            gaps = [{
                "id": "01",
                "limitation": first_paragraphs(answer, 3),
                "evidence": "",
                "why": "",
                "direction": "",
            }]

    recommendations = []
    if kind == "recommendations":
        # Parse the LLM's structured recommendation output first.
        # recommendation_prompt produces numbered entries like:
        #   **1. Topic / Title**
        #   - **Why relevant:** ...
        #   - **What to look for:** ...
        import re as _rre
        llm_recs: list[dict] = []
        rec_blocks = _rre.split(r"\n\*\*(\d+)\.\s+", answer)
        # rec_blocks: [preamble, "1", block1, "2", block2, ...]
        items_r = rec_blocks[1:]
        while len(items_r) >= 2:
            rank_str, block_text = items_r[0], items_r[1]
            items_r = items_r[2:]
            title_m = _rre.match(r"([^\n*]+)", block_text.strip())
            title = title_m.group(1).strip().rstrip("*").strip() if title_m else f"Recommendation {rank_str}"
            why_m = _rre.search(r"\*\*Why relevant:\*\*\s*(.*?)(?=\n\s*-\s*\*\*|\Z)", block_text, _rre.DOTALL)
            look_m = _rre.search(r"\*\*What to look for:\*\*\s*(.*?)(?=\n\s*-\s*\*\*|\Z)", block_text, _rre.DOTALL)
            src_m  = _rre.search(r"\*\*Source in evidence:\*\*\s*(.*?)(?=\n\s*-\s*\*\*|\Z)", block_text, _rre.DOTALL)
            why    = (why_m.group(1).strip()  if why_m  else "").replace("\n", " ")
            look   = (look_m.group(1).strip() if look_m else "").replace("\n", " ")
            src    = (src_m.group(1).strip()  if src_m  else "").replace("\n", " ")
            if title and len(title) < 300:
                llm_recs.append({
                    "rank":  int(rank_str),
                    "title": title,
                    "why":   why or look or src or NOT_REPORTED,
                    "url":   None,
                })

        if llm_recs:
            recommendations = llm_recs
        else:
            # Fallback: arXiv papers if any were found
            for i, paper in enumerate(papers[:5], start=1):
                recommendations.append({
                    "rank":  i,
                    "title": paper.title,
                    "why":   (paper.abstract or "")[:280] or NOT_REPORTED,
                    "url":   paper.url,
                })
            if not recommendations:
                rec_body = _section(sections, "recommended reading", "recommend") or answer
                recommendations.append({
                    "rank":  1,
                    "title": "From the local collection",
                    "why":   first_paragraphs(rec_body, 3),
                    "url":   None,
                })

    landscape: list[dict[str, Any]] = []
    if kind == "overview":
        by_kind: dict[str, list[str]] = {}
        for cite in citations:
            topic = cite.section or cite.kind or "Literature"
            by_kind.setdefault(topic, [])
            label = cite.title or cite.filename or cite.tag
            if label not in by_kind[topic]:
                by_kind[topic].append(label)
        if not by_kind and papers:
            by_kind["Discovered papers"] = [p.title for p in papers[:8]]
        landscape = [{"topic": k, "papers": v} for k, v in by_kind.items()]

    return {
        "kind": kind,
        "task_label": task_label(kind),
        "quick_take": quick,
        "findings": findings,
        "why_it_matters": why,
        "agree": _section(sections, "where they agree", "agree"),
        "differ": _section(sections, "where they differ", "differ"),
        "improves": _section(sections, "improves", "what paper"),
        "remaining_gap": _section(sections, "remaining gap", "open gap"),
        "bottom_line": _section(sections, "bottom line"),
        "comparison": comparison,
        "paper_comparison": paper_rows,
        "gaps": gaps,
        "recommendations": recommendations,
        "landscape": landscape,
        "sections": sections,
        "tabs": tabs_for(kind),
        "stats": {
            "sources_searched": indexed_docs,
            "relevant_sources": len(citations),
            "chunks_used": len(chunks or []),
        },
    }
