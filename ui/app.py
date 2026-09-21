"""Research Assistant — multi-page Streamlit UI.

Pages
-----
Research   — main search, question answering, RAG trace inspector
Evaluate   — multi-model eval runner, comparison table, charts
System     — architecture diagram, pipeline walkthrough, live config
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("RESEARCH_API_URL", "http://localhost:8000").rstrip("/")
QUERY_TIMEOUT = httpx.Timeout(connect=8.0, read=900.0, write=30.0, pool=10.0)
INGEST_TIMEOUT = httpx.Timeout(connect=8.0, read=600.0, write=30.0, pool=10.0)
CSS_PATH = Path(__file__).with_name("workspace.css")

# ── Action quick-start prompts ────────────────────────────────────────────────
ACTIONS = [
    ("Summarize paper", "Summarize the key findings, methodology, and contributions of the papers in my local library."),
    ("Compare papers", "Compare the papers in my library: problem, method, dataset, key finding, and limitation."),
    ("Find gaps", "What remains unresolved in this literature, and what research directions follow?"),
    ("Explain concept", "Explain retrieval-augmented generation using the papers I have indexed."),
    ("Find evidence", "What evidence in my library supports or challenges dense retrieval for academic QA?"),
    ("Recommend", "Recommend the next papers I should read on citation-grounded question answering."),
    ("Literature map", "Give a literature landscape of the topics covered in my local collection."),
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Outfit:wght@400;500;600;700&display=swap">',
    unsafe_allow_html=True,
)
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "page": "Research",
    "threads": [],
    "active": None,
    "history": [],
    "last_response": None,
    "last_query": "",
    "suggestions": [],
    "suggest_partial": "",
    "query_model": os.getenv("OLLAMA_MODEL", "llama3:8b"),
    "installed_models": [],
    "_last_submitted_model": os.getenv("OLLAMA_MODEL", "llama3:8b"),
    # Compare-papers flow
    "compare_mode": False,        # True while paper-selection UI is shown
    # Paper-picker flow (used by Find Gaps, Summarize, etc.)
    "paper_pick_mode": False,     # True while single-paper picker is shown
    "paper_pick_action": "",      # which action triggered the picker
    "compare_selected": [],       # list of selected filenames
    # Eval custom question / guardrail / output tests
    "_cq_last": None,
    "_gr_last": None,
    "_ot_last": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── API helpers ────────────────────────────────────────────────────────────────

def _get(path: str, timeout: float = 8.0) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, payload: dict, timeout: httpx.Timeout = QUERY_TIMEOUT) -> dict | None:
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        raise exc


def get_health() -> dict | None:
    return _get("/health", 4.0)


def get_library() -> dict:
    return _get("/library") or {"papers": [], "count": 0}


def get_models() -> list[str]:
    data = _get("/models")
    if data:
        return data.get("models") or []
    return []


def get_suggestions(partial: str, titles: list[str]) -> list[str]:
    if len(partial.strip()) < 3:
        return []
    try:
        r = httpx.post(
            f"{API_URL}/suggest",
            json={"partial": partial, "context_titles": titles},
            timeout=4.0,
        )
        r.raise_for_status()
        return r.json().get("suggestions") or []
    except Exception:
        return []


def run_query(text: str, model: str | None = None,
              paper_filter: list[str] | None = None) -> dict:
    payload: dict = {
        "query": text,
        "history": st.session_state.history[-8:],
        "force_local_rag": True,
    }
    if model:
        payload["model"] = model
    if paper_filter:
        payload["paper_filter"] = paper_filter
    r = httpx.post(f"{API_URL}/query", json=payload, timeout=QUERY_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_eval_results() -> dict:
    return _get("/eval/results", 20.0) or {}


def thread_title(query: str) -> str:
    cleaned = " ".join(query.strip().split())
    if len(cleaned) <= 52:
        return cleaned.rstrip(" ?") or "Untitled inquiry"
    return cleaned[:49].rstrip() + "…"


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _esc(v) -> str:
    return html.escape(str(v))


def html_table(headers: list, rows: list) -> str:
    if not headers:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def render_sources(citations: list) -> None:
    st.markdown('<div class="source-rail"><h3>Sources</h3></div>', unsafe_allow_html=True)
    if not citations:
        st.caption("Nothing cited yet — run a search.")
        return
    for cite in citations:
        num = f"{int(cite.get('source_id') or 0):02d}"
        title = _esc(cite.get("title") or cite.get("filename") or "Untitled")
        loc_bits = []
        if cite.get("page"):
            loc_bits.append(f"p. {cite['page']}")
        if cite.get("section"):
            loc_bits.append(str(cite["section"]))
        loc = _esc(" · ".join(loc_bits) if loc_bits else str(cite.get("locator") or ""))
        snippet = _esc((cite.get("snippet") or "")[:220])
        st.markdown(
            f'<div class="src"><span class="num">{num}</span> <span class="ttl">{title}</span>'
            f'<div class="meta">{loc}</div>'
            f'<div class="meta">{snippet}</div></div>',
            unsafe_allow_html=True,
        )
        with st.expander("View evidence"):
            if cite.get("filename"):
                st.caption(f"File · {cite['filename']}")
            st.write(cite.get("snippet") or "")
            if cite.get("url"):
                st.caption(cite["url"])


def render_overview(pres: dict, answer: str) -> None:
    kind = pres.get("kind")

    # ── Answer status badge ───────────────────────────────────────────────
    _STATUS_LABELS = {
        "research_grounded":            ("RESEARCH-GROUNDED",           "status-badge-research"),
        "general_explanation":          ("GENERAL EXPLANATION",          "status-badge-general"),
        "partial_evidence":             ("PARTIAL EVIDENCE",             "status-badge-partial"),
        "no_supporting_evidence":       ("NO SUPPORTING EVIDENCE",       "status-badge-none"),
        "external_information_required":("EXTERNAL INFO REQUIRED",       "status-badge-realtime"),
        "out_of_scope":                 ("OUT OF SCOPE",                 "status-badge-scope"),
        "model_unavailable":            ("MODEL UNAVAILABLE",            "status-badge-error"),
    }
    raw_status = pres.get("answer_status") or ""
    if raw_status in _STATUS_LABELS:
        label, cls = _STATUS_LABELS[raw_status]
        st.markdown(
            f'<div class="answer-status-badge {cls}">{label}</div>',
            unsafe_allow_html=True,
        )

    # ── Quick take heading ─────────────────────────────────────────────────
    quick = pres.get("quick_take") or ""
    if quick:
        st.markdown('<div class="section-kicker">Quick take</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="quick-take">{_esc(quick)}</p>', unsafe_allow_html=True)

    if kind == "gaps":
        st.markdown(
            '<div class="gap-flow"><span>Established</span><span>What papers solve</span>'
            "<span>Unresolved</span><span>Gap</span><span>Direction</span></div>",
            unsafe_allow_html=True,
        )
    # "findings" is shown only when it is a short distinct summary, not the full answer body
    _findings = pres.get("findings") or ""
    if _findings and len(_findings) < 350 and _findings.strip() != (quick or "").strip():
        st.markdown("**From the paper:**")
        st.markdown(_findings)
    if pres.get("why_it_matters") and kind not in {"analysis"}:
        st.markdown("##### Why it matters")
        st.markdown(pres["why_it_matters"])
    if pres.get("bottom_line") and kind not in {"comparison", "paper_comparison"}:
        st.markdown("##### Bottom line")
        st.markdown(pres["bottom_line"])

    # ── Main answer content ────────────────────────────────────────────────
    if kind == "factual":
        import re as _re
        sections_local = pres.get("sections") or {}
        answer_section = sections_local.get("answer") or ""
        evidence_section = sections_local.get("supporting evidence") or ""

        if answer_section:
            # New format: render Answer body (quick_take already showed first line)
            qt = (quick or "").strip()
            ans_clean = answer_section.strip()
            if qt and ans_clean.startswith(qt[:60]):
                remainder = ans_clean[len(qt):].strip()
                if remainder:
                    st.markdown(remainder)
            else:
                st.markdown(ans_clean)
            if evidence_section:
                st.markdown("---")
                st.markdown("**From the research paper:**")
                st.markdown(evidence_section.strip())
        else:
            # Old/fallback format: strip any heading that duplicates quick_take,
            # then render the body ONCE.
            _body = answer
            # Remove Quick Take block (already rendered above)
            _body = _re.sub(r"###\s+Quick Take.*?(?=\n###|\Z)", "", _body,
                            flags=_re.DOTALL | _re.IGNORECASE).strip()
            # Remove "What the paper says about this" heading line (keep content)
            _body = _re.sub(r"^###\s+What the paper says about this\s*\n",
                            "", _body, flags=_re.MULTILINE | _re.IGNORECASE)
            # Remove "General context" heading line (keep content)
            _body = _re.sub(r"^###\s+General context\s*\n",
                            "\n**General context:** ", _body, flags=_re.MULTILINE | _re.IGNORECASE)
            # Strip any duplicated paragraph (exact same content appearing twice)
            paragraphs = [p.strip() for p in _re.split(r"\n\s*\n", _body) if p.strip()]
            seen: set[str] = set()
            unique_paragraphs: list[str] = []
            for para in paragraphs:
                key = para[:120].lower()
                if key not in seen:
                    seen.add(key)
                    unique_paragraphs.append(para)
            _deduped = "\n\n".join(unique_paragraphs).strip()
            if _deduped:
                st.markdown(_deduped)
            elif not quick and answer:
                st.markdown(answer)

    elif not (pres.get("findings") or pres.get("why_it_matters") or pres.get("bottom_line") or quick):
        if answer:
            st.markdown(answer)

    if kind == "analysis":
        if pres.get("why_it_matters"):
            st.markdown("##### Conclusion / Significance")
            st.markdown(pres["why_it_matters"])
        elif not quick and answer:
            import re as _re
            parts = [p.strip() for p in _re.split(r"\n\s*\n", answer) if p.strip()]
            st.markdown("\n\n".join(parts[:2]) if parts else answer)


def render_comparison(pres: dict) -> None:
    paper = pres.get("paper_comparison")
    table = pres.get("comparison")
    if paper and paper.get("rows"):
        st.markdown("##### Paper comparison")
        st.markdown(html_table(paper.get("headers") or [], paper.get("rows") or []), unsafe_allow_html=True)
    if table and table.get("rows"):
        st.markdown("##### Method comparison")
        st.markdown(html_table(table.get("headers") or [], table.get("rows") or []), unsafe_allow_html=True)
        if table.get("incomplete"):
            st.caption("Empty cells = Not reported — sources did not support a full comparison.")
    bl = (table or {}).get("bottom_line") or pres.get("bottom_line")
    if bl:
        st.markdown("##### Bottom line"); st.markdown(bl)
    for label, key in [("Where they agree", "agree"), ("Where they differ", "differ"),
                       ("What one improves", "improves"), ("Remaining gap", "remaining_gap")]:
        if pres.get(key):
            st.markdown(f"##### {label}"); st.markdown(pres[key])


def render_gaps(pres: dict) -> None:
    gaps = pres.get("gaps") or []
    if not gaps:
        st.info("No research gaps were identified from the available evidence.")
        return
    for gap in gaps:
        gid = _esc(str(gap.get("id", "")))
        limitation = (gap.get("limitation") or "").strip()
        evidence   = (gap.get("evidence")   or "").strip()
        why        = (gap.get("why")        or "").strip()
        direction  = (gap.get("direction")  or "").strip()

        # Build the card body — show every non-empty field
        body_parts = []
        if limitation:
            body_parts.append(f'<p><strong>Limitation:</strong> {_esc(limitation)}</p>')
        if evidence:
            body_parts.append(f'<p><strong>Evidence:</strong> {_esc(evidence)}</p>')
        if why:
            body_parts.append(f'<p><strong>Why it matters:</strong> {_esc(why)}</p>')
        if direction:
            body_parts.append(f'<p><strong>Possible direction:</strong> {_esc(direction)}</p>')
        body_html = "".join(body_parts) or f"<p>{_esc(limitation)}</p>"

        st.markdown(
            f'<div class="gap-card">'
            f'<div class="gid">GAP {gid}</div>'
            f'{body_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_recs(pres: dict) -> None:
    recs = pres.get("recommendations") or []
    if not recs:
        st.info("No recommendations could be generated from the available evidence.")
        return
    for rec in recs:
        title = _esc(str(rec.get("title") or "Untitled"))
        why   = _esc(str(rec.get("why")   or ""))
        rank  = _esc(str(rec.get("rank",  "")))
        url   = rec.get("url") or ""
        url_html = f'<a href="{_esc(url)}" target="_blank">Open paper ↗</a>' if url else ""
        st.markdown(
            f'<div class="rec-card">'
            f'<div class="gid">READ NEXT · {rank}</div>'
            f'<strong>{title}</strong>'
            f'<p>{why}</p>'
            f'{("<p>" + url_html + "</p>") if url_html else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_landscape(pres: dict) -> None:
    st.markdown("##### Research landscape")
    for group in pres.get("landscape") or []:
        papers = "".join(f"<li>{_esc(str(p))}</li>" for p in group.get("papers") or [])
        st.markdown(
            f'<div class="land-card"><strong>{_esc(str(group.get("topic")))}</strong>'
            f"<ul>{papers}</ul></div>",
            unsafe_allow_html=True,
        )
    if pres.get("why_it_matters"):
        st.markdown("##### How the areas relate"); st.markdown(pres["why_it_matters"])


def render_rag_trace(result: dict) -> None:
    """Show the full RAG pipeline: KB stats + query-level trace."""
    inspector = result.get("inspector") or {}
    traces = result.get("traces") or []
    chunks = inspector.get("retrieved_chunks") or []
    llm_meta = inspector.get("llm") or {}
    lifecycle = inspector.get("lifecycle") or {}
    orchestration = result.get("orchestration") or {}
    guardrail = result.get("guardrail") or {}
    telemetry = result.get("telemetry") or {}

    # ── 1. Pipeline step row ──────────────────────────────────────────────
    st.markdown('<div class="pipe-label">Pipeline</div>', unsafe_allow_html=True)
    step_labels = [
        ("guardrail_input",  "Input guard"),
        ("planner",          "Intent plan"),
        ("model_selection",  "Model select"),
        ("rag_search",       "RAG retrieve"),
        ("arxiv_search",     "arXiv"),
        ("research_qa",      "LLM answer"),
        ("analysis",         "LLM answer"),
        ("comparison",       "LLM answer"),
        ("gap_analysis",     "LLM answer"),
        ("recommendation",   "LLM answer"),
        ("realtime_fallback","LLM answer"),
        ("guardrail_output", "Output guard"),
    ]
    seen_answer = False
    steps_html = ""
    for tool, label in step_labels:
        trace = next((t for t in traces if t.get("tool") == tool), None)
        if trace is None:
            continue
        if label == "LLM answer":
            if seen_answer:
                continue
            seen_answer = True
        status = trace.get("status", "ok")
        cls = "pipe-step-ok" if status == "ok" else ("pipe-step-warn" if status == "fallback" else "pipe-step-err")
        elapsed = trace.get("elapsed_ms")
        elapsed_str = f" {elapsed:.0f}ms" if elapsed else ""
        steps_html += f'<div class="pipe-step {cls}"><span>{label}</span><small>{elapsed_str}</small></div>'
        steps_html += '<div class="pipe-arrow">→</div>'
    if steps_html.endswith('<div class="pipe-arrow">→</div>'):
        steps_html = steps_html[: -len('<div class="pipe-arrow">→</div>')]
    st.markdown(f'<div class="pipe-row">{steps_html}</div>', unsafe_allow_html=True)

    # ── 2. Knowledge Base statistics ─────────────────────────────────────
    # Draw from kb_stats (new) with graceful fallback to old inspector fields
    kb = inspector.get("kb_stats") or inspector.get("chunking") and {
        "total_chunks": (inspector.get("chunking") or {}).get("total_chunks_in_index", "—"),
        "pdf_count": "—",
        "sources": [],
        "chunking": inspector.get("chunking") or {},
        "embedding": inspector.get("embedding") or {},
        "vector_store": (inspector.get("matching") or {}).get("dense_backend", "—"),
    } or {}

    if kb:
        st.markdown("#### Knowledge base")
        chunking = kb.get("chunking") or {}
        embedding = kb.get("embedding") or {}
        kb_rows = [
            ["Documents (PDFs)", str(kb.get("pdf_count", "—"))],
            ["Total chunks", str(kb.get("total_chunks", "—"))],
            ["Chunk size (tokens)", str(chunking.get("chunk_size_tokens", "—"))],
            ["Chunk overlap (tokens)", str(chunking.get("overlap_tokens", "—"))],
            ["Chunk step (tokens)", str(chunking.get("step_tokens", "—"))],
            ["Chunking method", str(chunking.get("method", "—"))],
            ["Embedding model", str(embedding.get("model", "—"))],
            ["Embedding dimension", str(embedding.get("dimension") or "—")],
            ["Vector store", str(kb.get("vector_store", "—"))],
        ]
        st.markdown(html_table(["Parameter", "Value"], kb_rows), unsafe_allow_html=True)

        # Per-source breakdown
        sources = kb.get("sources") or []
        if sources:
            src_rows = [
                [s.get("filename", "—"),
                 str(s.get("chunks", "—")),
                 str(s.get("pages_extracted", "—")),
                 str(s.get("source_kind", "—"))]
                for s in sources
            ]
            st.markdown(html_table(
                ["Source", "Chunks", "Pages extracted", "Type"],
                src_rows,
            ), unsafe_allow_html=True)

    # ── 3. Current query retrieval ────────────────────────────────────────
    st.markdown("#### Current query retrieval")
    model_used = (
        telemetry.get("model")
        or (orchestration.get("selected_model") if isinstance(orchestration, dict) else None)
        or lifecycle.get("selected_model")
        or "—"
    )
    query_type = lifecycle.get("query_type") or "—"
    answer_status = lifecycle.get("answer_status") or (
        (result.get("presentation") or {}).get("answer_status") or "—"
    )
    grounding_score = (
        orchestration.get("grounding_score") if isinstance(orchestration, dict) else None
    )
    elapsed = telemetry.get("elapsed_ms", 0)

    n_retrieved = len(chunks)
    n_passed = min(n_retrieved, 5)   # RERANK_TOP_K default; actual value from config
    # Try to read actual rerank top-k from matching telemetry
    matching = inspector.get("matching") or {}
    if matching.get("k_requested"):
        n_passed = int(matching.get("k_requested"))

    query_rows = [
        ["Query type",         query_type],
        ["Chunks retrieved",   str(n_retrieved)],
        ["Chunks passed to LLM", str(n_passed if n_retrieved > 0 else 0)],
        ["Model used",         model_used],
        ["Answer status",      answer_status],
        ["Grounding score",    f"{grounding_score:.0%}" if grounding_score is not None else "—"],
        ["Grounding flagged",  "Yes ⚠" if (orchestration.get("grounding_flagged") if isinstance(orchestration, dict) else False) else "No ✓"],
        ["Response latency",   f"{elapsed:.0f} ms"],
    ]
    st.markdown(html_table(["Metric", "Value"], query_rows), unsafe_allow_html=True)

    # ── 4. Retrieved chunk detail ─────────────────────────────────────────
    if chunks:
        st.markdown("**Retrieved context** — ranked by RRF score")
        for i, ch in enumerate(chunks[:8], 1):
            loc = []
            if ch.get("page"):
                loc.append(f"p. {ch['page']}")
            if ch.get("section"):
                loc.append(str(ch["section"]))
            label = f"[{i}] {ch.get('filename') or 'Source'}"
            if loc:
                label += "  ·  " + " · ".join(loc)
            score = ch.get("score") or ch.get("rrf_score")
            if score is not None:
                label += f"  ·  RRF {float(score):.4f}"
            with st.expander(label):
                st.write(ch.get("text") or "")
                meta = ch.get("origins") or ch.get("metadata") or {}
                if meta:
                    st.caption(str(meta))
    else:
        st.info("No chunks retrieved for this query — answer is based on general knowledge or a direct response.")

    # ── 5. LLM prompt inspection ──────────────────────────────────────────
    if llm_meta.get("raw_prompt"):
        with st.expander("Raw prompt sent to model"):
            st.code(llm_meta["raw_prompt"][:3000], language="text")

    # ── 6. Resource usage ─────────────────────────────────────────────────
    res_after = inspector.get("resources_after") or {}
    if res_after and res_after.get("rss_mb"):
        r1, r2, r3 = st.columns(3)
        r1.metric("RSS MB", f"{res_after.get('rss_mb', '—')}")
        r2.metric("CPU %", f"{res_after.get('process_cpu_percent', '—')}")
        gpu = res_after.get("gpu") or {}
        r3.metric("GPU MB", str(gpu.get("memory_used_mb") or "N/A"))


def render_answer_area(result: dict) -> None:
    """Render the main answer with intent-aware tabs + source rail."""
    pres = result.get("presentation") or {}
    kind = pres.get("kind", "factual")
    answer = result.get("answer", "")
    citations = result.get("citations") or []
    telemetry = result.get("telemetry") or {}
    orchestration = result.get("orchestration") or {}

    # ── Sanity guard: never render a completely blank answer ──────────────
    if not (answer or "").strip():
        answer = (
            "The model did not return a response. "
            "Check that Ollama is running and the selected model is available."
        )

    # Stats row — includes active model
    elapsed = telemetry.get("elapsed_ms", 0)
    model_used = (
        telemetry.get("model")
        or orchestration.get("selected_model")
        or "—"
    )
    st.markdown(
        f"""<div class="stat-row">
          <div class="stat"><span>Task</span><strong>{_esc(pres.get("task_label","Evidence review"))}</strong></div>
          <div class="stat"><span>Sources cited</span><strong>{len(citations)}</strong></div>
          <div class="stat"><span>Model · Latency</span><strong>{_esc(model_used)} · {elapsed:.0f} ms</strong></div>
        </div>""",
        unsafe_allow_html=True,
    )

    main_col, src_col = st.columns([2.35, 0.9])

    with src_col:
        render_sources(citations)

    with main_col:
        tabs_labels = pres.get("tabs") or ["Answer", "RAG Trace", "Sources"]
        # Always inject a RAG Trace tab
        if "RAG Trace" not in tabs_labels:
            tabs_labels = tabs_labels + ["RAG Trace"]

        tabs = st.tabs(tabs_labels)
        tab_map = {label: tab for label, tab in zip(tabs_labels, tabs)}

        with tab_map.get("Answer", tab_map.get("Overview", tabs[0])):
            st.markdown('<div class="paper">', unsafe_allow_html=True)
            render_overview(pres, answer)
            st.markdown("</div>", unsafe_allow_html=True)

        if "Comparison" in tab_map:
            with tab_map["Comparison"]:
                st.markdown('<div class="paper">', unsafe_allow_html=True)
                render_comparison(pres)
                st.markdown("</div>", unsafe_allow_html=True)

        if "Research Gaps" in tab_map:
            with tab_map["Research Gaps"]:
                st.markdown('<div class="paper">', unsafe_allow_html=True)
                render_gaps(pres)
                st.markdown("</div>", unsafe_allow_html=True)

        if "Recommendations" in tab_map:
            with tab_map["Recommendations"]:
                st.markdown('<div class="paper">', unsafe_allow_html=True)
                render_recs(pres)
                st.markdown("</div>", unsafe_allow_html=True)

        if "Landscape" in tab_map:
            with tab_map["Landscape"]:
                st.markdown('<div class="paper">', unsafe_allow_html=True)
                render_landscape(pres)
                st.markdown("</div>", unsafe_allow_html=True)

        if "Analysis" in tab_map:
            with tab_map["Analysis"]:
                st.markdown('<div class="paper">', unsafe_allow_html=True)
                sections = pres.get("sections") or {}

                # Render the seven canonical analysis sections in order.
                # Use a priority-ordered lookup so slight LLM heading variations still match.
                _SECTION_ORDER = [
                    ("research problem",                 "Research Problem"),
                    ("methodology / approach",           "Methodology / Approach"),
                    ("methodology",                      "Methodology / Approach"),
                    ("approach",                         "Methodology / Approach"),
                    ("key findings / results",           "Key Findings / Results"),
                    ("key findings",                     "Key Findings / Results"),
                    ("findings",                         "Key Findings / Results"),
                    ("contributions",                    "Contributions"),
                    ("limitations",                      "Limitations"),
                    ("conclusion / significance",        "Conclusion / Significance"),
                    ("conclusion",                       "Conclusion / Significance"),
                    ("significance",                     "Conclusion / Significance"),
                    ("why it matters",                   "Conclusion / Significance"),
                ]
                rendered_headings: set[str] = set()
                rendered_any = False
                for section_key, display_heading in _SECTION_ORDER:
                    val = sections.get(section_key, "")
                    if val and val.strip() and display_heading not in rendered_headings:
                        st.markdown(f"##### {display_heading}")
                        st.markdown(val)
                        rendered_headings.add(display_heading)
                        rendered_any = True

                # Hard fallback: if no recognised sections found, show full answer
                if not rendered_any:
                    st.markdown(answer)
                st.markdown("</div>", unsafe_allow_html=True)

        if "Evidence" in tab_map:
            with tab_map["Evidence"]:
                inspector = result.get("inspector") or {}
                raw_chunks = inspector.get("retrieved_chunks") or []
                if raw_chunks:
                    for ch in raw_chunks:
                        loc = []
                        if ch.get("page"):
                            loc.append(f"p. {ch['page']}")
                        if ch.get("section"):
                            loc.append(str(ch["section"]))
                        label = f"{ch.get('filename') or 'Source'}"
                        if loc:
                            label += "  ·  " + " · ".join(loc)
                        with st.expander(label):
                            st.write(ch.get("text") or "")
                else:
                    for cite in citations:
                        with st.expander(f"{cite.get('tag')} {cite.get('title')}"):
                            st.write(cite.get("snippet") or "")

        if "Sources" in tab_map:
            with tab_map["Sources"]:
                for cite in citations:
                    st.markdown(
                        f"**{cite.get('tag')}** {_esc(cite.get('title',''))}  \n"
                        f"{_esc(cite.get('locator',''))}  \n"
                        f"*{_esc((cite.get('snippet') or '')[:200])}*"
                    )
                    if cite.get("url"):
                        st.caption(cite["url"])

        with tab_map.get("RAG Trace", tabs[-1]):
            render_rag_trace(result)


# ── Evaluate page charts ──────────────────────────────────────────────────────

def _bar_chart(data: dict[str, float], title: str, color: str = "#e8b84a", fmt: str = ".3f") -> None:
    """Render a simple horizontal bar chart using only st.progress + markdown."""
    if not data:
        st.caption(f"No data for {title}.")
        return
    st.markdown(f"**{title}**")
    max_val = max(data.values(), default=1) or 1
    for model, val in sorted(data.items(), key=lambda x: -x[1]):
        pct = int(val / max_val * 100)
        label = f"{val:{fmt}}"
        bar_html = (
            f'<div class="bar-row">'
            f'<span class="bar-model">{_esc(model)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="bar-val">{label}</span>'
            f"</div>"
        )
        st.markdown(bar_html, unsafe_allow_html=True)


def render_eval_charts(charts: dict) -> None:
    if not charts:
        st.info("Run an evaluation to generate charts.")
        return

    st.markdown("#### Accuracy & quality")
    c1, c2 = st.columns(2)
    with c1:
        _bar_chart(charts.get("bar_correctness") or {}, "Correctness (keyword coverage)", "#6ee7b7")
    with c2:
        _bar_chart(charts.get("bar_retrieval") or {}, "Retrieval quality", "#60a5fa")

    st.markdown("#### Hallucination & grounding")
    c3, c4 = st.columns(2)
    with c3:
        # Lower hallucination = better → invert color cue
        _bar_chart(charts.get("bar_hallucination") or {}, "Hallucination rate ↓ lower is better", "#ff8b6a")
    with c4:
        _bar_chart(charts.get("bar_latency") or {}, "Latency ms ↓ lower is better", "#a78bfa", fmt=".0f")

    st.markdown("#### Speed & resources")
    c5, c6 = st.columns(2)
    with c5:
        _bar_chart(charts.get("tokens_per_sec") or {}, "Tokens / sec ↑ higher is better", "#fbbf24", fmt=".2f")
    with c6:
        _bar_chart(charts.get("rss_mb") or {}, "Memory RSS MB ↓ lower is better", "#f472b6", fmt=".1f")

    # Scatter: latency vs correctness
    scatter = charts.get("scatter") or []
    if len(scatter) > 1:
        st.markdown("#### Quality vs speed trade-off")
        rows_html = ""
        for pt in scatter:
            rows_html += (
                f"<tr><td>{_esc(pt['model'])}</td>"
                f"<td>{pt.get('latency_ms',0):.0f} ms</td>"
                f"<td>{pt.get('correctness',0):.3f}</td>"
                f"<td>{pt.get('hallucination',0):.3f}</td>"
                f"<td>{pt.get('retrieval',0):.3f}</td></tr>"
            )
        st.markdown(
            html_table(
                ["Model", "Latency ms", "Correctness", "Hallucination", "Retrieval"],
                [[pt["model"], f"{pt.get('latency_ms',0):.0f}", f"{pt.get('correctness',0):.3f}",
                  f"{pt.get('hallucination',0):.3f}", f"{pt.get('retrieval',0):.3f}"]
                 for pt in scatter],
            ),
            unsafe_allow_html=True,
        )

    # Category breakdown
    cat_breakdown = charts.get("category_breakdown") or {}
    if cat_breakdown:
        st.markdown("#### Correctness by question category")
        # Gather all categories
        all_cats: list[str] = []
        for model_cats in cat_breakdown.values():
            for cat in model_cats:
                if cat not in all_cats:
                    all_cats.append(cat)
        models = list(cat_breakdown.keys())
        headers = ["Category"] + models
        rows = []
        for cat in all_cats:
            row = [cat]
            for model in models:
                val = cat_breakdown[model].get(cat)
                row.append(f"{val:.3f}" if val is not None else "—")
            rows.append(row)
        st.markdown(html_table(headers, rows), unsafe_allow_html=True)

    # Radar summary table
    radar = charts.get("radar") or {}
    if radar:
        st.markdown("#### Overall model scorecard")
        metrics = ["Correctness", "Relevance", "Retrieval", "Grounding", "Speed"]
        headers = ["Model"] + metrics
        rows = []
        for model, scores in radar.items():
            rows.append([model] + [f"{scores.get(m, 0):.3f}" for m in metrics])
        st.markdown(html_table(headers, rows), unsafe_allow_html=True)
        st.caption(
            "All metrics normalised 0–1. Grounding = 1 − hallucination_rate. "
            "Speed = 1 − (latency / max_latency). Higher is better for all columns."
        )


# ── Architecture / System page ────────────────────────────────────────────────

def render_architecture_page(health: dict | None) -> None:
    st.markdown("## System architecture")
    st.markdown(
        "This page shows the live architecture of the application — "
        "which services exist, how they connect, and what the current runtime state is."
    )

    # Pipeline diagram
    st.markdown("### Request pipeline")
    st.markdown(
        """<div class="arch-pipeline">
  <div class="arch-layer user-layer">
    <div class="arch-title">UI / Client</div>
    <div class="arch-boxes">
      <div class="arch-box">Streamlit UI<br><small>ui/app.py</small></div>
    </div>
  </div>
  <div class="arch-arrow-v">↓ HTTP</div>
  <div class="arch-layer api-layer">
    <div class="arch-title">API / Gateway</div>
    <div class="arch-boxes">
      <div class="arch-box">FastAPI<br><small>api/main.py</small></div>
      <div class="arch-box">Rate limiter<br><small>middleware</small></div>
      <div class="arch-box">Guardrails<br><small>guardrail_service</small></div>
    </div>
  </div>
  <div class="arch-arrow-v">↓</div>
  <div class="arch-layer orch-layer">
    <div class="arch-title">Orchestration</div>
    <div class="arch-boxes">
      <div class="arch-box">ResearchOrchestrator<br><small>orchestrator.py</small></div>
      <div class="arch-box">Complexity classifier<br><small>model_orchestration_service</small></div>
      <div class="arch-box">Intent planner<br><small>planner_service</small></div>
    </div>
  </div>
  <div class="arch-arrow-v">↓</div>
  <div class="arch-layer svc-layer">
    <div class="arch-title">Application services</div>
    <div class="arch-boxes">
      <div class="arch-box">ResearchQA<br><small>research_qa_service</small></div>
      <div class="arch-box">Analysis<br><small>analysis_service</small></div>
      <div class="arch-box">Comparison<br><small>comparison_service</small></div>
      <div class="arch-box">GapAnalysis<br><small>gap_analysis_service</small></div>
      <div class="arch-box">Recommendation<br><small>recommendation_service</small></div>
    </div>
  </div>
  <div class="arch-arrow-v">↓</div>
  <div class="arch-layer data-layer">
    <div class="arch-title">Data / Knowledge services</div>
    <div class="arch-boxes">
      <div class="arch-box">RAG service<br><small>rag_service.py</small></div>
      <div class="arch-box">Embedding<br><small>vector_db/embeddings.py</small></div>
      <div class="arch-box">FAISS / Chroma<br><small>vector_db/</small></div>
      <div class="arch-box">arXiv discovery<br><small>discovery_service.py</small></div>
      <div class="arch-box">PDF library<br><small>library_service.py</small></div>
    </div>
  </div>
  <div class="arch-arrow-v">↓</div>
  <div class="arch-layer llm-layer">
    <div class="arch-title">LLM service</div>
    <div class="arch-boxes">
      <div class="arch-box">OllamaLLM<br><small>llm/llm_model.py</small></div>
      <div class="arch-box">Prompt builder<br><small>llm/prompt_builder.py</small></div>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # Live config
    if health:
        st.markdown("### Live configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                html_table(
                    ["Setting", "Value"],
                    [
                        ["Ollama host", health.get("ollama_host", "—")],
                        ["Default model", health.get("ollama_model", "—")],
                        ["Ollama reachable", "✓ Yes" if health.get("ollama_reachable") else "✗ No"],
                        ["Vector backend", health.get("vector_backend", "—")],
                        ["Indexed chunks", str(health.get("indexed_chunks", "—"))],
                        ["PDFs in library", str(health.get("pdf_count", "—"))],
                        ["Embedding model", health.get("embedding_model", "—")],
                    ],
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                html_table(
                    ["Guardrail / rate limit", "Value"],
                    [
                        ["Rate limiting", "Enabled (30 req / 60 s on /query, /ingest, /eval/run)"],
                        ["Input scope check", "Keyword + off-topic pattern matching"],
                        ["Output grounding", "Sentence–context token overlap (threshold 0.35)"],
                        ["Model routing", "Complexity heuristic → lightweight or capable model"],
                    ],
                ),
                unsafe_allow_html=True,
            )

    # Service responsibility table
    st.markdown("### Service responsibilities")
    st.markdown(
        html_table(
            ["Service", "File", "Responsibility"],
            [
                ["Application service", "services/research_qa_service.py … recommendation_service.py",
                 "User-facing task logic: QA, analysis, comparison, gap detection, recommendations"],
                ["RAG / Retrieval service", "services/rag_service.py",
                 "PDF + code ingestion, chunking, hybrid dense+BM25 search, RRF reranking"],
                ["LLM service", "llm/llm_model.py + llm/prompt_builder.py",
                 "Ollama SDK wrapper, prompt templates, token stats, graceful failure"],
                ["Embedding service", "vector_db/embeddings.py",
                 "Sentence-transformer embeddings (all-MiniLM-L6-v2), batch + query encoding"],
                ["Data / Knowledge service", "services/library_service.py + services/discovery_service.py",
                 "PDF library management, arXiv Atom API discovery"],
                ["Vector DB", "vector_db/faiss_db.py + vector_db/chroma_db.py",
                 "FAISS (IndexFlatL2) and Chroma backends, telemetry, runtime switching"],
                ["API / Gateway", "api/main.py + api/schemas.py",
                 "FastAPI endpoints, rate limiting middleware, request/response contracts"],
                ["Orchestration", "services/orchestrator.py",
                 "End-to-end pipeline: guardrail → plan → model select → retrieve → generate → validate"],
                ["Guardrails", "services/guardrail_service.py",
                 "Input scope check, output grounding validation, off-topic detection"],
                ["Model orchestration", "services/model_orchestration_service.py",
                 "Complexity classification, model tier selection, grounding score"],
                ["Evaluation", "eval/runner.py + eval/metrics.py + eval/analyze.py",
                 "Multi-model eval, 7 deterministic metrics, chart data, markdown report"],
                ["Suggest service", "services/suggest_service.py",
                 "Heuristic prompt suggestions from partial queries, no LLM call"],
            ],
        ),
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════ #
#  SIDEBAR                                                                    #
# ════════════════════════════════════════════════════════════════════════════ #

health = get_health()
ready = bool(health and health.get("status") == "ok")

with st.sidebar:
    # Status pill
    if ready:
        st.markdown('<div class="live-pill">Live · library connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">API offline</span>', unsafe_allow_html=True)
        st.caption(f"{API_URL}/health")

    # Page navigation
    st.markdown("**Navigate**")
    for page_name in ["Research", "Evaluate", "System"]:
        active_cls = "nav-active" if st.session_state.page == page_name else ""
        if st.button(page_name, key=f"nav-{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

    st.markdown("---")

    # Research page extras only
    if st.session_state.page == "Research":
        st.markdown("**Sessions**")
        if st.button("New inquiry", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_response = None
            st.session_state.last_query = ""
            st.session_state.active = None
            st.session_state.suggestions = []
            st.rerun()
        for i, thread in enumerate(st.session_state.threads):
            label = thread.get("title") or f"Inquiry {i + 1}"
            if st.button(label, key=f"thread-{i}", use_container_width=True):
                st.session_state.active = i
                st.session_state.history = thread.get("history") or []
                st.session_state.last_response = thread.get("last_response")
                st.session_state.last_query = thread.get("query") or ""
                st.rerun()

        st.markdown("**Research library**")
        lib = get_library()
        papers = lib.get("papers") or []
        st.caption(f"{lib.get('count', len(papers))} papers indexed")
        search = st.text_input("Filter library", placeholder="filename…")
        filtered = [p for p in papers if not search or search.lower() in p.get("filename", "").lower()]
        for paper in filtered:
            status = paper.get("status")
            mark_cls = "ok" if status == "indexed" else "warn"
            mark = "Indexed" if status == "indexed" else "Needs re-index"
            pages = paper.get("pages") or "—"
            st.markdown(
                f'<div class="lib-card"><strong>{_esc(str(paper.get("filename")))}</strong>'
                f'<div class="{mark_cls}">{pages} pages · {mark}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Remove", key=f"rm-{paper.get('filename')}"):
                try:
                    httpx.delete(f"{API_URL}/library/{paper.get('filename')}", timeout=20.0).raise_for_status()
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        uploaded = st.file_uploader("Add PDF", type=["pdf"])
        if uploaded is not None and st.button("Save to library"):
            try:
                httpx.post(
                    f"{API_URL}/library/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=60.0,
                ).raise_for_status()
                st.success("Saved. Rebuild the index to search it.")
            except Exception as exc:
                st.error(str(exc))

        if st.button("Rebuild index", use_container_width=True):
            try:
                ingest = httpx.post(f"{API_URL}/ingest", json={"rebuild": True}, timeout=INGEST_TIMEOUT)
                ingest.raise_for_status()
                st.success("Index rebuilt.")
            except Exception as exc:
                st.error(str(exc))

        # Model selector — populated from /models endpoint
        st.markdown("**Model**")
        installed = get_models()
        if installed:
            st.session_state.installed_models = installed
        model_options = st.session_state.installed_models or [os.getenv("OLLAMA_MODEL", "llama3:8b")]
        current = st.session_state.query_model
        if current not in model_options:
            model_options = [current] + model_options
        chosen = st.selectbox(
            "Active model",
            options=model_options,
            index=model_options.index(current),
            key="model_selectbox",
        )
        # Detect model change: update session state immediately
        if chosen != st.session_state.query_model:
            st.session_state.query_model = chosen
        st.caption("Orchestration routes complex queries to the most capable model available.")


# ════════════════════════════════════════════════════════════════════════════ #
#  HERO (all pages)                                                           #
# ════════════════════════════════════════════════════════════════════════════ #

st.markdown(
    f"""<div class="hero">
  <div style="display:flex;gap:0.85rem;align-items:center;">
    <div class="hero-mark">R</div>
    <div>
      <h1>Research Assistant</h1>
      <p>Literature intelligence workspace — compare, cite, and find the gap.</p>
    </div>
  </div>
  <div class="live-pill">{"Online" if ready else "Waiting for API"}</div>
</div>""",
    unsafe_allow_html=True,
)

# ── Page navigation pills ──────────────────────────────────────────────────────
pg_cols = st.columns([1, 1, 1, 5])
for col, name in zip(pg_cols[:3], ["Research", "Evaluate", "System"]):
    active = st.session_state.page == name
    cls = "page-pill-active" if active else "page-pill"
    if col.button(name, key=f"pgbtn-{name}"):
        st.session_state.page = name
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════ #
#  PAGE: RESEARCH                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

if st.session_state.page == "Research":

    # ── Action quick-starts ───────────────────────────────────────────────────
    # "Compare papers" → multi-paper selection UI
    # "Find gaps", "Summarize paper" → single-paper picker when >1 paper indexed
    # All other buttons fire directly.

    # Actions that need a paper picker when multiple papers are indexed
    _SINGLE_PAPER_ACTIONS = {"Summarize paper", "Find gaps", "Explain concept"}

    if not st.session_state.compare_mode and not st.session_state.paper_pick_mode:
        row1 = st.columns(4)
        row2 = st.columns(3)
        for i, (label, prompt) in enumerate(ACTIONS):
            target = row1[i] if i < 4 else row2[i - 4]
            if target.button(label, use_container_width=True):
                if label == "Compare papers":
                    st.session_state.compare_mode = True
                    st.session_state.compare_selected = []
                    st.rerun()
                elif label in _SINGLE_PAPER_ACTIONS:
                    # If more than one paper, let user pick which one
                    _lib_check = get_library()
                    _pdfs_check = [p.get("filename","") for p in (_lib_check.get("papers") or []) if p.get("filename")]
                    if len(_pdfs_check) > 1:
                        st.session_state.paper_pick_mode = True
                        st.session_state.paper_pick_action = label
                        st.session_state["_paper_pick_prompt"] = prompt
                        st.rerun()
                    else:
                        # Only one paper — just fire directly with paper_filter
                        st.session_state["pending_query"] = prompt
                        if _pdfs_check:
                            st.session_state["pending_paper_filter"] = _pdfs_check
                else:
                    st.session_state["pending_query"] = prompt

    # ── Single-paper picker (Summarize / Find gaps / Explain concept) ─────────
    if st.session_state.paper_pick_mode:
        _lib_pick = get_library()
        _all_pick = [p.get("filename","") for p in (_lib_pick.get("papers") or []) if p.get("filename")]
        _action_label = st.session_state.paper_pick_action
        _action_prompt = st.session_state.get("_paper_pick_prompt", "")

        st.markdown(f"### {_action_label} — select paper")
        st.caption("Choose which paper to use. Only the selected paper's content will be used.")

        _pick_selected = st.radio(
            "Paper",
            options=_all_pick,
            format_func=lambda x: f"📄 {x}",
            key="paper_pick_radio",
        )

        col_go2, col_cancel2 = st.columns([1, 5])
        with col_go2:
            if st.button(f"▶ {_action_label}", type="primary", key="pick_go"):
                st.session_state["pending_query"] = _action_prompt
                st.session_state["pending_paper_filter"] = [_pick_selected]
                st.session_state.paper_pick_mode = False
                st.session_state.paper_pick_action = ""
                st.rerun()
        with col_cancel2:
            if st.button("✕ Cancel", key="pick_cancel"):
                st.session_state.paper_pick_mode = False
                st.session_state.paper_pick_action = ""
                st.rerun()
        st.divider()

    # ── Paper-selection UI (Compare Papers) ───────────────────────────────────
    if st.session_state.compare_mode:
        _lib_cmp = get_library()
        _all_pdfs = [p.get("filename", "") for p in (_lib_cmp.get("papers") or []) if p.get("filename")]

        st.markdown("### Compare papers")
        if not _all_pdfs:
            st.warning("No research papers are indexed yet. Upload PDFs first, then rebuild the index.")
            if st.button("Cancel", key="cmp_cancel_empty"):
                st.session_state.compare_mode = False
                st.rerun()
        else:
            st.caption(
                f"**{len(_all_pdfs)} paper(s) in library.** "
                "Tick at least 2 papers. Only the ticked papers will be used as comparison evidence."
            )

            _selected = []
            for pdf_name in _all_pdfs:
                checked = pdf_name in st.session_state.compare_selected
                if st.checkbox(f"📄 {pdf_name}", value=checked, key=f"cmp_{pdf_name}"):
                    _selected.append(pdf_name)
            st.session_state.compare_selected = _selected

            _n = len(_selected)
            if _n >= 2:
                st.success(f"**{_n} papers selected:** " + " · ".join(f"`{n}`" for n in _selected))
            elif _n == 1:
                st.info(f"1 paper selected: `{_selected[0]}` — select at least one more.")
            else:
                st.info("Select 2 or more papers to enable comparison.")

            col_go, col_cancel = st.columns([1, 5])
            with col_go:
                _can_compare = _n >= 2
                if st.button(
                    f"▶ Compare {_n} papers" if _can_compare else "Compare",
                    type="primary",
                    disabled=not _can_compare,
                    key="cmp_go",
                ):
                    _names = ", ".join(f'"{n}"' for n in _selected)
                    _cmp_query = (
                        f"Compare these {_n} research papers: {_names}. "
                        "For each paper provide: paper title, research problem, objective, "
                        "method/approach, model/architecture, dataset/benchmarks, "
                        "evaluation metrics, key results with actual numbers, "
                        "main contributions, limitations, and future work. "
                        "Then provide a comparison analysis: what they have in common, "
                        "how their methods differ, results compared, strengths and weaknesses, "
                        "key trade-offs, and what research gap emerges from comparing them."
                    )
                    st.session_state["pending_query"] = _cmp_query
                    st.session_state["pending_paper_filter"] = list(_selected)
                    st.session_state.compare_mode = False
                    st.session_state.compare_selected = []
                    st.rerun()
            with col_cancel:
                if st.button("✕ Cancel", key="cmp_cancel"):
                    st.session_state.compare_mode = False
                    st.session_state.compare_selected = []
                    st.rerun()
        st.divider()

    # Query input
    lib = get_library()
    pdf_titles = [p.get("filename", "") for p in (lib.get("papers") or [])]

    query_input = st.text_area(
        "Research question",
        placeholder="Ask a literature question — compare methods, find gaps, or request evidence.",
        height=90,
        label_visibility="collapsed",
        key="query_input_main",
    )

    # Dynamic suggestions
    current_partial = (query_input or "").strip()
    if current_partial and current_partial != st.session_state.suggest_partial and len(current_partial) >= 3:
        st.session_state.suggest_partial = current_partial
        st.session_state.suggestions = get_suggestions(current_partial, pdf_titles)

    suggestions = st.session_state.suggestions
    if suggestions and current_partial:
        st.markdown('<div class="suggest-header">Suggested questions</div>', unsafe_allow_html=True)
        sug_cols = st.columns(min(len(suggestions), 3))
        for i, suggestion in enumerate(suggestions[:5]):
            col = sug_cols[i % len(sug_cols)]
            if col.button(suggestion[:72] + ("…" if len(suggestion) > 72 else ""), key=f"sug-{i}"):
                st.session_state["pending_query"] = suggestion
                st.session_state.suggestions = []
                st.rerun()

    pending = st.session_state.pop("pending_query", None)
    _paper_filter = st.session_state.pop("pending_paper_filter", None)
    effective = (pending or query_input or "").strip()

    # ── Submit row: Search button + inline model picker ───────────────────
    _submit_col, _model_col = st.columns([3, 2])
    with _submit_col:
        # Empty label spacer so the button aligns with the selectbox bottom
        st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
        submit = st.button("Search literature", type="primary", use_container_width=True)
    with _model_col:
        _installed_inline = st.session_state.installed_models or [os.getenv("OLLAMA_MODEL", "llama3:8b")]
        _cur = st.session_state.query_model
        if _cur not in _installed_inline:
            _installed_inline = [_cur] + _installed_inline
        _chosen_inline = st.selectbox(
            "Model",
            options=_installed_inline,
            index=_installed_inline.index(_cur),
            key="model_inline",
            label_visibility="visible",
        )
        if _chosen_inline != st.session_state.query_model:
            st.session_state.query_model = _chosen_inline

    # Model-change re-run: if the user switched model and there's a previous
    # query, offer a one-click re-run button so they can compare.
    current_model = st.session_state.query_model
    last_model = st.session_state.get("_last_submitted_model", current_model)
    model_changed = (
        current_model != last_model
        and st.session_state.last_query
        and not submit
        and not pending
    )
    if model_changed:
        st.info(
            f"Model changed from **{last_model}** → **{current_model}**. "
            "Click **Re-run with new model** to compare answers."
        )
        if st.button("Re-run with new model", key="rerun_model"):
            pending = st.session_state.last_query
            effective = pending

    if submit or pending:
        if not effective:
            st.warning("Enter a research question.")
        else:
            status_box = st.status("Understanding your question", expanded=True)
            try:
                status_box.update(label="Searching your papers", state="running")
                status_box.write("Ranking relevant evidence…")
                result = run_query(effective, model=st.session_state.query_model,
                                   paper_filter=_paper_filter)
                status_box.update(label="Preparing the answer", state="complete")

                # Record which model was actually used
                st.session_state["_last_submitted_model"] = st.session_state.query_model

                st.session_state.history.append({"role": "user", "content": effective})
                st.session_state.history.append({"role": "assistant", "content": result.get("answer", "")})
                st.session_state.last_response = result
                st.session_state.last_query = effective
                st.session_state.suggestions = []

                title = thread_title(effective)
                if st.session_state.active is None:
                    st.session_state.threads.append(
                        {"title": title, "query": effective,
                         "history": list(st.session_state.history), "last_response": result}
                    )
                    st.session_state.active = len(st.session_state.threads) - 1
                else:
                    st.session_state.threads[st.session_state.active] = {
                        "title": title if len(st.session_state.history) <= 2
                        else st.session_state.threads[st.session_state.active]["title"],
                        "query": effective,
                        "history": list(st.session_state.history),
                        "last_response": result,
                    }
            except httpx.ConnectError:
                status_box.update(label="Could not reach the API", state="error")
                st.error(f"Start the API at {API_URL}")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    status_box.update(label="Rate limit hit", state="error")
                    st.warning("Too many requests — please wait a moment before trying again.")
                else:
                    status_box.update(label="Search failed", state="error")
                    st.error(str(exc))
            except Exception as exc:
                status_box.update(label="Search failed", state="error")
                st.error(str(exc))

    result = st.session_state.last_response
    if not result:
        st.markdown(
            """<div class="empty">
  <h2>Open a brief. Leave with evidence.</h2>
  <p>Search your PDFs like a research partner would: sharp question, ranked sources, comparison when it matters, gaps when the literature runs out.</p>
  <div class="empty-grid">
    <div class="empty-card"><b>Compare</b>Methods, papers, trade-offs — as a table.</div>
    <div class="empty-card"><b>Evidence</b>Every claim sits next to a page, section, and excerpt.</div>
    <div class="empty-card"><b>Gaps</b>What is solved, what is missing, what to study next.</div>
    <div class="empty-card"><b>Library</b>Upload PDFs, index them, and search across your collection.</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        render_answer_area(result)


# ════════════════════════════════════════════════════════════════════════════ #
#  PAGE: EVALUATE                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

elif st.session_state.page == "Evaluate":
    st.markdown("## Model evaluation")
    st.markdown(
        "Run the built-in evaluation on **33 representative questions** across 16 categories "
        "and all three models — same questions, same knowledge base, same prompts. "
        "Results are measured with deterministic metrics, not LLM-judge scores. "
        "Progress is saved after every question so a laptop sleep/restart loses nothing."
    )

    # Metric definitions
    with st.expander("How metrics are calculated"):
        st.markdown(
            """
| Metric | Definition |
|---|---|
| **Correctness** | Fraction of `gold_keywords` found in the answer (case-insensitive). Citation penalty (×0.7) if citation required but missing. |
| **Relevance** | Jaccard token overlap between question and answer. |
| **Retrieval quality** | File-path hit rate for items with `expected_files`; otherwise keyword coverage of retrieved chunk text. |
| **Hallucination rate** | Fraction of answer sentences with fewer than 4 content-token overlaps with retrieved context. Lower is better. |
| **Test-pass rate** | Fraction of code-generation items whose extracted Python passes the deterministic unit test. |
| **Latency** | End-to-end `orchestrator.run()` wall-clock milliseconds. |
| **Tokens** | Ollama `prompt_eval_count + eval_count` (estimated as `len/4` when unavailable). |
| **Resources** | Process RSS MB and CPU % via psutil; GPU memory via nvidia-smi when available. |
"""
        )

    # Run controls
    eval_col1, eval_col2 = st.columns([3, 1])
    with eval_col1:
        models_raw = st.text_input(
            "Models (comma-separated Ollama tags)",
            value="llama3:8b,codellama:7b,starcoder2:3b",
        )
    with eval_col2:
        limit = st.number_input("Max questions", min_value=1, max_value=33, value=33)

    run_eval = st.button("Run evaluation", type="primary")

    # Status polling
    status_data = _get("/eval/status") or {}
    if status_data.get("running"):
        phase = status_data.get("phase", "running")
        done = status_data.get("done", 0)
        total = status_data.get("total", 1) or 1
        model_now = status_data.get("model", "")
        item_now = status_data.get("item_id", "")
        items_per_model = status_data.get("items_per_model", total)
        model_idx = status_data.get("current_model_index", 0)
        n_models = max(1, total // max(items_per_model, 1))
        # Show per-model progress
        model_done = done % max(items_per_model, 1)
        pct = int(done / total * 100)
        progress_text = (
            f"{phase} — Model {model_idx+1}/{n_models}: {model_now} · "
            f"Q{model_done+1}/{items_per_model} ({item_now}) — overall {done}/{total}"
        )
        st.progress(pct / 100, text=progress_text)
        st.info("Evaluation running. Refresh the page to check progress. Results will appear when complete.")
        time.sleep(3)
        st.rerun()

    if run_eval:
        models_list = [m.strip() for m in models_raw.split(",") if m.strip()]
        try:
            resp = httpx.post(
                f"{API_URL}/eval/run",
                json={"models": models_list, "limit": int(limit), "disable_arxiv": True},
                timeout=30.0,
            )
            resp.raise_for_status()
            st.success("Evaluation started. This page will refresh when complete.")
            time.sleep(1)
            st.rerun()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                st.warning("An evaluation run is already in progress.")
            else:
                st.error(str(exc))
        except Exception as exc:
            st.error(str(exc))

    # Results
    results = get_eval_results()
    if results.get("available"):
        skipped = results.get("models_skipped") or []
        if skipped:
            st.warning(f"Models not installed (skipped): {', '.join(skipped)}")

        by_model = results.get("by_model") or {}

        # Summary comparison table
        st.markdown("### Summary comparison")
        summaries = {
            m: b["summary"]
            for m, b in by_model.items()
            if isinstance(b, dict) and b.get("summary")
        }
        if summaries:
            metric_keys = [
                ("correctness_mean", "Correctness"),
                ("relevance_mean", "Relevance"),
                ("retrieval_quality_mean", "Retrieval"),
                ("hallucination_rate_mean", "Hallucination ↓"),
                ("test_pass_rate", "Test pass"),
                ("latency_ms_mean", "Latency ms ↓"),
                ("tokens_mean", "Tokens"),
                ("tokens_per_sec_mean", "tok/s"),
                ("rss_mb_mean", "RSS MB"),
                ("cpu_percent_mean", "CPU %"),
            ]
            headers = ["Model"] + [m[1] for m in metric_keys]
            rows = []
            for model, s in summaries.items():
                row = [model] + [
                    str(s.get(k) if s.get(k) is not None else "—")
                    for k, _ in metric_keys
                ]
                rows.append(row)
            st.markdown(html_table(headers, rows), unsafe_allow_html=True)

        # Charts
        st.markdown("### Visual analysis")
        charts = results.get("charts") or {}
        render_eval_charts(charts)

        # ── Category-wise metric breakdown (per-category expanders) ─────────
        st.markdown("### Category-wise evaluation results")
        st.caption(
            "Click ▶ next to a category to see bar charts comparing all models across "
            "all metrics. Hallucination ↓ and Latency ↓ — lower is better."
        )
        _cat_full = charts.get("category_full_breakdown") or {}
        if _cat_full:
            # Canonical category display names and descriptions
            _CAT_META = {
                "code_explanation":      ("Code Explanation",        "Questions asking what a specific function/module does"),
                "code_retrieval":        ("Code Retrieval",          "Questions asking which file/component handles a task"),
                "dependency_understanding": ("Dependency Understanding", "Questions about which components depend on each other"),
                "bug_analysis":          ("Bug Analysis",            "Questions asking what could cause incorrect behaviour"),
                "code_generation":       ("Code Generation",         "Questions asking the model to write code"),
                "refactoring":           ("Refactoring",             "Questions asking for improvement suggestions"),
                "rag_qa":                ("RAG-based QA",            "Questions answered from the indexed research papers"),
                "analysis":              ("Paper Analysis",          "Deep analysis of methodology and structure"),
                "comparison":            ("Paper Comparison",        "Side-by-side comparison of papers"),
                "gap_analysis":          ("Gap Analysis",            "Finding open research problems"),
                "recommendation":        ("Recommendation",          "Suggesting next papers to read"),
                "discovery":             ("Literature Discovery",    "Finding related literature"),
                "repo_multi_file":       ("Repository Multi-file",   "Questions spanning multiple source files"),
                "repo_retrieval":        ("Repo Retrieval",          "Specific retrieval from the codebase"),
                "hallucination_probe":   ("Hallucination Probe",     "Testing whether the model fabricates facts"),
            }

            # Collect all categories in results
            _all_cats_set: list[str] = []
            for _m_cats in _cat_full.values():
                for _cat in _m_cats:
                    if _cat not in _all_cats_set:
                        _all_cats_set.append(_cat)
            # Sort: canonical order first, then alphabetical for unknown
            _CANONICAL_ORDER = list(_CAT_META.keys())
            _all_cats_set.sort(
                key=lambda c: (_CANONICAL_ORDER.index(c) if c in _CANONICAL_ORDER else 999, c)
            )

            _models = list(_cat_full.keys())

            _METRICS = [
                ("correctness",        "Correctness",     "#6ee7b7", True),
                ("relevance",          "Relevance",       "#60a5fa", True),
                ("retrieval_quality",  "Retrieval",       "#a78bfa", True),
                ("hallucination_rate", "Hallucination ↓", "#ff8b6a", False),  # lower=better → invert
                ("latency_ms",         "Latency ms ↓",   "#fbbf24", False),
            ]

            # Summary table: all categories × all models, correctness column
            _sum_headers = ["Category", "n"] + _models
            _sum_rows = []
            for _cat in _all_cats_set:
                _display = _CAT_META.get(_cat, (_cat.replace("_", " ").title(), ""))[0]
                _n = next(
                    (int(_cat_full[_m].get(_cat, {}).get("n", 0))
                     for _m in _models if _cat in _cat_full.get(_m, {})),
                    0,
                )
                _row = [_display, str(_n)]
                for _m in _models:
                    _v = (_cat_full.get(_m) or {}).get(_cat, {}).get("correctness")
                    _row.append(f"{_v:.3f}" if _v is not None else "—")
                _sum_rows.append(_row)

            st.markdown("**Correctness overview — all categories**")
            st.markdown(html_table(_sum_headers, _sum_rows), unsafe_allow_html=True)
            st.markdown("---")

            # Per-category expanders with full metric bar charts
            st.markdown("**Per-category detail — click ▶ to expand**")
            for _cat in _all_cats_set:
                _display, _desc = _CAT_META.get(_cat, (_cat.replace("_", " ").title(), ""))
                # Show correctness values as a quick summary in the expander header
                _corr_preview = "  ·  ".join(
                    f"{_m}: {(_cat_full.get(_m) or {}).get(_cat, {}).get('correctness', 0):.2f}"
                    for _m in _models
                    if (_cat_full.get(_m) or {}).get(_cat)
                )
                _n_q = next(
                    (int(_cat_full[_m].get(_cat, {}).get("n", 0))
                     for _m in _models if _cat in _cat_full.get(_m, {})),
                    0,
                )
                with st.expander(f"▶  {_display}  ({_n_q} questions)  —  Correct: {_corr_preview}"):
                    if _desc:
                        st.caption(_desc)

                    # Full metrics table for this category
                    _tbl_headers = ["Metric", "Direction"] + _models
                    _tbl_rows = []
                    for _mk, _ml, _color, _higher in _METRICS:
                        _direction_label = "↑ higher" if _higher else "↓ lower"
                        _tr = [_ml, _direction_label]
                        for _m in _models:
                            _v = (_cat_full.get(_m) or {}).get(_cat, {}).get(_mk)
                            _tr.append(f"{_v:.3f}" if _v is not None else "—")
                        _tbl_rows.append(_tr)
                    st.markdown(html_table(_tbl_headers, _tbl_rows), unsafe_allow_html=True)

                    # Bar charts: one per metric, all models side by side
                    for _mk, _ml, _color, _higher in _METRICS:
                        st.markdown(f"**{_ml}**")
                        # Gather values for this metric across all models
                        _vals = {
                            _m: (_cat_full.get(_m) or {}).get(_cat, {}).get(_mk)
                            for _m in _models
                        }
                        _vals = {m: v for m, v in _vals.items() if v is not None}
                        if not _vals:
                            st.caption("No data for this metric in this category.")
                            continue
                        _max_v = max(_vals.values()) or 1
                        for _m, _v in _vals.items():
                            # For lower-is-better metrics, invert bar so shorter = better visually
                            _bar_pct = int((_v / _max_v) * 100) if _higher else int((1 - _v / max(_max_v, 1)) * 100)
                            # Use fill pct for display but show actual value
                            _fill_pct = int((_v / _max_v) * 100)
                            _bar = (
                                f'<div class="bar-row">'
                                f'<span class="bar-model">{_esc(_m)}</span>'
                                f'<div class="bar-track">'
                                f'<div class="bar-fill" style="width:{_fill_pct}%;background:{_color}"></div>'
                                f'</div>'
                                f'<span class="bar-val">{_v:.3f}</span>'
                                f'</div>'
                            )
                            st.markdown(_bar, unsafe_allow_html=True)
                    st.markdown("")

            # Best / worst model summary
            if len(_models) > 1:
                st.markdown("---")
                st.markdown("**Best model per category — correctness**")
                _best_rows = []
                for _cat in _all_cats_set:
                    _display = _CAT_META.get(_cat, (_cat.replace("_", " ").title(), ""))[0]
                    _scores = {
                        _m: (_cat_full.get(_m) or {}).get(_cat, {}).get("correctness")
                        for _m in _models
                    }
                    _scores = {m: v for m, v in _scores.items() if v is not None}
                    if not _scores:
                        continue
                    _best_m  = max(_scores, key=lambda x: _scores[x])
                    _worst_m = min(_scores, key=lambda x: _scores[x])
                    _gap = round(_scores[_best_m] - _scores[_worst_m], 3)
                    _best_rows.append([
                        _display,
                        f"{_best_m} ({_scores[_best_m]:.3f})",
                        f"{_worst_m} ({_scores[_worst_m]:.3f})",
                        f"{_gap:.3f}",
                    ])
                if _best_rows:
                    st.markdown(
                        html_table(["Category", "Best model", "Weakest model", "Gap"], _best_rows),
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Run an evaluation to see category-wise results. Results will appear here after the run.")

        # ── Category interpretations ─────────────────────────────────────────
        _cat_interps = charts.get("category_interpretations") or {}
        if _cat_interps:
            st.markdown("### Why models perform differently per category")
            st.caption("Rule-based interpretation of each model's performance on every question category.")
            for _cat in _all_cats_set if _cat_full else sorted(_cat_interps.keys()):
                _display = _CAT_META.get(_cat, (_cat.replace("_", " ").title(), ""))[0] if _cat_full else _cat
                _overall_text = (_cat_interps.get(_cat) or {}).get("_overall", "")
                _model_texts  = {m: t for m, t in (_cat_interps.get(_cat) or {}).items() if m != "_overall"}
                if not _model_texts:
                    continue
                with st.expander(f"📊  {_display} — interpretation"):
                    if _overall_text:
                        st.markdown(_overall_text)
                        st.markdown("---")
                    for _m, _txt in _model_texts.items():
                        st.markdown(f"**{_m}**")
                        st.markdown(_txt)
                        st.markdown("")

        # ── Custom question evaluator ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Ask your own question — evaluate all models")
        st.caption(
            "Type any question and evaluate how all 3 models answer it on the same knowledge base. "
            "Each answer is scored with output tests (grounding, relevance, citation, hallucination)."
        )
        _cq_models_raw = st.text_input(
            "Models to evaluate",
            value="llama3:8b,codellama:7b,starcoder2:3b",
            key="cq_models",
        )
        _custom_q = st.text_area(
            "Your question",
            placeholder="e.g. What is the main contribution of the indexed paper?",
            height=80,
            key="cq_input",
        )
        if st.button("Evaluate this question", type="primary", key="cq_run"):
            _cq = (_custom_q or "").strip()
            if not _cq:
                st.warning("Enter a question first.")
            else:
                _cq_models = [m.strip() for m in _cq_models_raw.split(",") if m.strip()]
                with st.spinner(f"Evaluating {len(_cq_models)} models… (this takes 1–3 min)"):
                    try:
                        _cq_resp = httpx.post(
                            f"{API_URL}/eval/ask",
                            json={"question": _cq, "models": _cq_models},
                            timeout=httpx.Timeout(connect=8.0, read=900.0, write=30.0, pool=10.0),
                        )
                        _cq_resp.raise_for_status()
                        _cq_data = _cq_resp.json()
                        st.session_state["_cq_last"] = _cq_data
                    except Exception as _exc:
                        st.error(f"Evaluation failed: {_exc}")
                        st.session_state["_cq_last"] = None

        _cq_data = st.session_state.get("_cq_last")
        if _cq_data:
            st.markdown(f"**Question:** {_cq_data.get('question','')}")
            _cq_summary = _cq_data.get("summary") or {}
            _cq_rows_raw = _cq_data.get("rows") or []

            # Summary metrics table
            _cq_hdr = ["Model", "Correct", "Relevant", "Retrieval", "Hallucin ↓", "Latency ms", "OT Pass rate", "OT Verdict"]
            _cq_tbl = []
            for _m, _s in _cq_summary.items():
                _ot = _s.get("output_tests") or {}
                _cq_tbl.append([
                    _m,
                    f"{_s.get('correctness', 0):.3f}",
                    f"{_s.get('relevance', 0):.3f}",
                    f"{_s.get('retrieval_quality', 0):.3f}",
                    f"{_s.get('hallucination_rate', 0):.3f}",
                    f"{_s.get('latency_ms', 0):.0f}",
                    f"{_ot.get('pass_rate', 0):.0%}",
                    _ot.get("verdict", "—"),
                ])
            st.markdown(html_table(_cq_hdr, _cq_tbl), unsafe_allow_html=True)

            # Per-model answer + output test detail
            for _row in _cq_rows_raw:
                _m = _row.get("model", "")
                _ot = (_cq_summary.get(_m) or {}).get("output_tests") or {}
                _explanation = (_cq_summary.get(_m) or {}).get("explanation") or ""
                _verdict_icon = "✓" if _ot.get("verdict") == "PASS" else ("⚠" if _ot.get("verdict") == "WARN" else "✗")
                with st.expander(f"{_verdict_icon}  {_m}  —  {_ot.get('verdict','—')}  ({_ot.get('pass_rate',0):.0%} tests passed)"):

                    # ── Why this model answered this way ─────────────────────
                    if _explanation:
                        st.markdown("#### Why this model performed this way")
                        st.markdown(_explanation)
                        st.markdown("---")

                    st.markdown("**Answer:**")
                    st.write((_row.get("answer") or "")[:1000])

                    # Output test results as mini table
                    _ot_rows = _ot.get("results") or []
                    if _ot_rows:
                        st.markdown("**Output tests:**")
                        _ot_tbl = []
                        for _t in _ot_rows:
                            _icon = "✓" if _t["passed"] else "✗"
                            _ot_tbl.append([
                                _icon + " " + _t["test_id"],
                                _t["name"],
                                _t["category"],
                                f"{_t['score']:.3f}",
                                _t["severity"],
                                _t["detail"][:80],
                            ])
                        st.markdown(
                            html_table(["", "Test", "Category", "Score", "Severity", "Detail"], _ot_tbl),
                            unsafe_allow_html=True,
                        )

                    # Bar chart: output test scores
                    st.markdown("**Test scores:**")
                    for _t in _ot_rows:
                        _clr = "#6ee7b7" if _t["passed"] else "#ff8b6a"
                        _pct = int(_t["score"] * 100)
                        st.markdown(
                            f'<div class="bar-row">'
                            f'<span class="bar-model">{_esc(_t["test_id"])} {_esc(_t["name"][:30])}</span>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{_pct}%;background:{_clr}"></div></div>'
                            f'<span class="bar-val">{_t["score"]:.3f}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ── Guardrail demo ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Guardrail evaluation — 5 categories × 5 questions × 3 models")
        st.caption(
            "25 guardrail questions across 5 categories. "
            "**Run guardrail tests** for instant input-level checks (no LLM needed). "
            "**Evaluate models on guardrails** runs all questions through all 3 models and scores each — takes ~5 min."
        )

        _gc_models_raw = st.text_input("Models", value="llama3:8b,codellama:7b,starcoder2:3b", key="gc_models")
        if st.button("Evaluate models on guardrails", type="primary", key="gr_eval_run"):
            _gc_mods = [m.strip() for m in _gc_models_raw.split(",") if m.strip()]
            with st.spinner(f"Evaluating {len(_gc_mods)} models × 30 questions… (~8 min)"):
                try:
                    _ge_resp = httpx.post(
                        f"{API_URL}/eval/guardrail-eval",
                        json={"models": _gc_mods},
                        timeout=httpx.Timeout(connect=8.0, read=900.0, write=30.0, pool=10.0),
                    )
                    _ge_resp.raise_for_status()
                    st.session_state["_ge_last"] = _ge_resp.json()
                    # Also run the fast input checks to populate _gr_last for the category cards
                    try:
                        _gr_resp = httpx.get(f"{API_URL}/eval/guardrail-tests", timeout=30.0)
                        _gr_resp.raise_for_status()
                        st.session_state["_gr_last"] = _gr_resp.json()
                    except Exception:
                        pass
                except Exception as _exc:
                    st.error(f"Evaluation failed: {_exc}")
                    st.session_state["_ge_last"] = None

        _gr_data = st.session_state.get("_gr_last")   # populated automatically after eval
        _ge_data = st.session_state.get("_ge_last")   # full model evaluation
        # Import question list for per-question detail rendering
        try:
            from eval.guardrail_tests import GUARDRAIL_QUESTIONS as _GQ_LIST
        except Exception:
            _GQ_LIST = []

        # ── Overall summary ───────────────────────────────────────────────────
        if _ge_data:
            st.markdown("**Model evaluation summary:**")
            _ge_models = _ge_data.get("models") or []
            _ge_bm = _ge_data.get("by_model") or {}
            _ge_tbl = [
                [m, f"{_ge_bm[m].get('safe',0)}/{_ge_bm[m].get('total',0)}", f"{_ge_bm[m].get('pass_rate',0):.0%}"]
                for m in _ge_models if m in _ge_bm
            ]
            if _ge_tbl:
                st.markdown(html_table(["Model", "Safe responses", "Pass rate"], _ge_tbl), unsafe_allow_html=True)

        st.markdown("---")

        # ── 5 category cards ──────────────────────────────────────────────────
        _GC_STATIC = [
            ("GC1", "🛡",  "Input Guardrails",                "#6ee7b7", "Injection, PII, toxicity, jailbreak, length"),
            ("GC2", "📤", "Output Guardrails",                "#60a5fa", "Grounding, blank answers, citation, refusal, format"),
            ("GC3", "🗄",  "Retrieval & Data",                "#a78bfa", "Empty index, wrong source, irrelevant context, citations, stale data"),
            ("GC4", "🤖", "Agent & Tool",                     "#fbbf24", "Tool misuse, loops, confirmation, scope, multi-step"),
            ("GC5", "⚖",  "Ethical, Brand & Legal",          "#f472b6", "Bias, copyright, harmful content, disclaimers, tone"),
            ("GC6", "⚙",  "Infrastructure & Operational",    "#fb923c", "Rate limiting, token budget, latency timeout, fallback, data residency"),
        ]

        for _gc_id, _gc_icon, _gc_name, _gc_color, _gc_desc in _GC_STATIC:
            # Fast check summary
            _by_type = (_gr_data or {}).get("by_type") or {}
            _gc_block = _by_type.get(_gc_id) or {}
            _tp = _gc_block.get("tests_passed", 0)
            _tt = _gc_block.get("tests_total", 0)
            _gv = _gc_block.get("verdict", "")

            # Full eval summary
            _cat_sum = ((_ge_data or {}).get("category_summaries") or {}).get(_gc_id) or {}
            _ge_models_list = (_ge_data or {}).get("models") or []

            # Build expander header
            _header_parts = [f"{_gc_icon}  **{_gc_name}**  — {_gc_desc}"]
            if _gr_data and _tt:
                _badge_c = "#6ee7b7" if _gv == "PASS" else ("#fbbf24" if _gv == "WARN" else "#ff8b6a")
                _header_parts.append(f"  |  Input checks: {_tp}/{_tt}")
            if _ge_data and _cat_sum:
                _bm = _cat_sum.get("best_model", "")
                _header_parts.append(f"  |  Best model: {_bm}")

            with st.expander("  ".join(_header_parts)):
                st.caption(_gc_desc)

                # ── Input-level results table ─────────────────────────────────
                _cases = _gc_block.get("cases") or []
                if _cases:
                    st.markdown("**Input guardrail checks (no LLM):**")
                    _ic_rows = []
                    for _c in _cases:
                        _v = "✓" if _c.get("test_passed") else "✗"
                        _ic_rows.append([
                            _v + " " + _c.get("id",""),
                            _c.get("name","")[:40],
                            _c.get("question","")[:55],
                            _c.get("detail","")[:60],
                        ])
                    st.markdown(html_table(["", "Check", "Input", "Result"], _ic_rows), unsafe_allow_html=True)

                # ── Multi-model comparison ────────────────────────────────────
                _ms = _cat_sum.get("model_summaries") or {}
                if _ms:
                    st.markdown("**Multi-model comparison (all 5 questions):**")
                    _mm_rows = []
                    for _m in _ge_models_list:
                        _m_data = _ms.get(_m) or {}
                        _mm_rows.append([
                            _m,
                            f"{_m_data.get('safe',0)}/{_m_data.get('total',0)}",
                            f"{_m_data.get('pass_rate',0):.0%}",
                            f"{_m_data.get('avg_grounding',0):.2f}",
                        ])
                    st.markdown(html_table(["Model", "Safe/Total", "Pass %", "Avg Grounding"], _mm_rows), unsafe_allow_html=True)

                    # Bar charts per model
                    st.markdown("**Pass rate comparison:**")
                    for _m in _ge_models_list:
                        _pr = (_ms.get(_m) or {}).get("pass_rate", 0)
                        _pct = int(_pr * 100)
                        _clr = "#6ee7b7" if _pct >= 80 else ("#fbbf24" if _pct >= 50 else "#ff8b6a")
                        st.markdown(
                            f'<div class="bar-row"><span class="bar-model">{_esc(_m)}</span>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{_pct}%;background:{_clr}"></div></div>'
                            f'<span class="bar-val">{_pct}%</span></div>',
                            unsafe_allow_html=True,
                        )

                    # Per-question per-model detail
                    st.markdown("**Per-question model responses:**")
                    _cat_questions = [q for q in _GQ_LIST if q["category_id"] == _gc_id]
                    for _q in _cat_questions:
                        _qid = _q["id"]
                        with st.expander(f"  {_qid}: {_q.get('name','')} — `{(_q.get('question',''))[:60]}`"):
                            _qc1, _qc2 = st.columns(2)
                            with _qc1:
                                st.markdown("**❌ Without guardrail:**")
                                st.markdown(f'<div style="background:#2a0a0a;border:1px solid #ff8b6a;border-radius:8px;padding:0.6rem;color:#ffcccc;font-size:0.82rem;">{_esc(_q.get("without_guardrail",""))}</div>', unsafe_allow_html=True)
                            with _qc2:
                                st.markdown("**✅ With guardrail:**")
                                st.markdown(f'<div style="background:#0a2a0a;border:1px solid #6ee7b7;border-radius:8px;padding:0.6rem;color:#ccffcc;font-size:0.82rem;">{_esc(_q.get("with_guardrail",""))}</div>', unsafe_allow_html=True)

                            # Model answer comparison
                            if _ms:
                                for _m in _ge_models_list:
                                    _m_results = (_ms.get(_m) or {}).get("results") or []
                                    _q_result = next((r for r in _m_results if r.get("id") == _qid), None)
                                    if _q_result:
                                        _safe_icon = "✓" if _q_result.get("safe") else "✗"
                                        st.markdown(f"**{_m}** — {_safe_icon} {_q_result.get('verdict','—')}  |  {_q_result.get('detail','')}")
                                        if _q_result.get("answer_excerpt"):
                                            st.caption(f"Answer: {_q_result['answer_excerpt'][:180]}")

                    # ── Recommendation ────────────────────────────────────────
                    _rec = _cat_sum.get("recommendation", "")
                    if _rec:
                        st.markdown("---")
                        st.markdown("**📌 Recommendation:**")
                        st.markdown(
                            f'<div style="background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.3);border-radius:10px;padding:0.8rem;color:#e2e8f0;">{_esc(_rec)}</div>',
                            unsafe_allow_html=True,
                        )
                elif not _gr_data and not _ge_data:
                    st.info("Click **Run guardrail tests** or **Evaluate models on guardrails** above to see results.")

        # ── Output testing against latest eval ───────────────────────────────
        st.markdown("---")
        st.markdown("### AI output testing — latest eval results")
        st.caption(
            "Runs 9 structured output tests against every answer in the latest evaluation. "
            "Tests check: blank answers, relevance, grounding, hallucination, citations, "
            "length, keyword coverage, and prompt leakage."
        )
        if st.button("Run output tests on latest eval", key="ot_run"):
            with st.spinner("Running output tests…"):
                try:
                    _ot_resp = httpx.get(f"{API_URL}/eval/output-tests", timeout=60.0)
                    _ot_resp.raise_for_status()
                    st.session_state["_ot_last"] = _ot_resp.json()
                except Exception as _exc:
                    st.error(f"Output tests failed: {_exc}")
                    st.session_state["_ot_last"] = None

        _ot_data = st.session_state.get("_ot_last")
        if _ot_data and _ot_data.get("available"):
            _ot_by_model = _ot_data.get("by_model") or {}
            # Summary table
            _ot_hdr = ["Model", "Total Qs", "Tests passed", "Tests failed", "Pass rate"]
            _ot_tbl = []
            for _m, _block in _ot_by_model.items():
                _n = _block.get("total", 0)
                _p = _block.get("passed", 0)
                _f = _block.get("failed", 0)
                _r = _block.get("pass_rate", 0)
                _ot_tbl.append([_m, str(_n), str(_p), str(_f), f"{_r:.0%}"])
            st.markdown(html_table(_ot_hdr, _ot_tbl), unsafe_allow_html=True)

            # Per-model drill-down
            for _m, _block in _ot_by_model.items():
                with st.expander(f"Output tests — {_m}  ({_block.get('pass_rate',0):.0%} pass rate)"):
                    _items = _block.get("items") or []
                    _item_tbl = []
                    for _item in _items:
                        _v = _item.get("verdict", "")
                        _icon = "✓" if _v == "PASS" else ("⚠" if _v == "WARN" else "✗")
                        _crit = ", ".join(_item.get("critical_failures") or []) or "—"
                        _item_tbl.append([
                            _item.get("id", ""),
                            _item.get("category", ""),
                            (_item.get("question") or "")[:55],
                            f"{_icon} {_v}",
                            f"{_item.get('pass_rate', 0):.0%}",
                            _crit[:50],
                        ])
                    st.markdown(
                        html_table(
                            ["ID", "Category", "Question", "Verdict", "Pass %", "Critical failures"],
                            _item_tbl,
                        ),
                        unsafe_allow_html=True,
                    )
        elif _ot_data and not _ot_data.get("available"):
            st.info("Run the main evaluation first, then click 'Run output tests on latest eval'.")

        # Per-model detail
        st.markdown("### Per-model item detail")
        model_names = list(by_model.keys())
        if model_names:
            selected_model = st.selectbox("Select model to inspect", model_names)
            block = by_model.get(selected_model) or {}
            items = block.get("items") or []
            if items:
                item_headers = ["ID", "Category", "Correct", "Relevant", "Retrieval", "Hallucin.", "Latency ms", "Answer excerpt"]
                item_rows = []
                for row in items:
                    m = row.get("metrics") or {}
                    item_rows.append([
                        row.get("id", ""),
                        row.get("category", ""),
                        f"{m.get('correctness', 0):.3f}",
                        f"{m.get('relevance', 0):.3f}",
                        f"{m.get('retrieval_quality', 0):.3f}",
                        f"{m.get('hallucination_rate', 0):.3f}",
                        f"{row.get('latency_ms', 0):.0f}",
                        (row.get("answer") or "")[:80] + "…",
                    ])
                st.markdown(html_table(item_headers, item_rows), unsafe_allow_html=True)

                # Retrieval trace drill-down
                st.markdown("#### RAG trace — retrieval → context → answer")
                st.caption(
                    "These traces show how retrieval quality drives answer quality. "
                    "Red = hallucinated despite having relevant context."
                )
                for row in items[:8]:
                    m = row.get("metrics") or {}
                    label = row.get("retrieval_label") or m.get("retrieval_label", "")
                    halluc = m.get("hallucinated_despite_context", False)
                    correct_enough = m.get("answer_correct_enough", False)
                    tag_cls = "trace-ok" if correct_enough and not halluc else "trace-warn"
                    tag = "✓ Grounded" if correct_enough and not halluc else ("⚠ Hallucinated" if halluc else "✗ Incorrect")
                    with st.expander(
                        f"{row.get('id')} · {row.get('category')} · {label} · {tag}"
                    ):
                        ctx = row.get("retrieved_context") or []
                        col_q, col_c, col_a = st.columns(3)
                        with col_q:
                            st.markdown("**Question**")
                            st.write(row.get("question", ""))
                        with col_c:
                            st.markdown("**Retrieved context (first chunk)**")
                            if ctx:
                                st.write((ctx[0].get("text") or "")[:400])
                            else:
                                st.caption("Nothing retrieved.")
                        with col_a:
                            st.markdown("**Answer**")
                            st.write((row.get("answer") or "")[:400])

        # Markdown analysis
        if results.get("analysis_markdown"):
            with st.expander("Full analysis report (Markdown)"):
                st.markdown(results["analysis_markdown"])
    else:
        st.info("No evaluation results yet. Run an evaluation above.")


# ════════════════════════════════════════════════════════════════════════════ #
#  PAGE: SYSTEM                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

elif st.session_state.page == "System":
    render_architecture_page(health)
