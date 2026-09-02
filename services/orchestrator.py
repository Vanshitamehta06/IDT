"""Async pipeline orchestrator: plan, execute tools, cite, synthesize.

Pipeline
--------
1. Guardrail – input scope check (realtime flagged but not hard-blocked)
2. Query-type classification (research / general / realtime / out_of_scope)
3. Complexity classification + model selection (preferred_model ALWAYS wins)
4. Planner – intent classification
5. RAG search (hybrid dense + BM25) — skipped for realtime/out_of_scope
6. arXiv discovery (optional)
7. Single specialist LLM call with query-type-aware prompt
8. Output grounding validation
9. Presentation assembly with answer_status field
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from api.schemas import (
    ArxivPaper,
    ChatTurn,
    ComplexityLevel,
    ExecutionPlan,
    GuardrailResult,
    IntentType,
    ModelOrchestrationDecision,
    QueryResponse,
    RetrievedChunk,
    TraceStep,
)
from config import GROUNDING_THRESHOLD, OLLAMA_MODEL
from llm.llm_model import OllamaLLM
from llm.prompt_builder import (
    bound_text,
    classify_query_type,
    format_history,
    general_explanation_prompt,
    out_of_scope_prompt,
    qa_prompt,
    realtime_fallback_prompt,
)
from services.analysis_service import AnalysisService
from services.citation_service import CitationService
from services.comparison_service import ComparisonService
from services.discovery_service import DiscoveryService
from services.gap_analysis_service import GapAnalysisService
from services.guardrail_service import check_input, check_output
from services.model_orchestration_service import (
    classify_complexity,
    select_model,
)
from services.planner_service import PlannerService, heuristic_plan
from services.presentation_service import build_presentation
from services.rag_service import RAGService
from services.recommendation_service import RecommendationService
from services.research_qa_service import ResearchQAService
from services.resource_monitor import snapshot_resources

# Answer status labels surfaced to the UI
ANSWER_STATUS_RESEARCH    = "research_grounded"
ANSWER_STATUS_GENERAL     = "general_explanation"
ANSWER_STATUS_PARTIAL     = "partial_evidence"
ANSWER_STATUS_NO_EVIDENCE = "no_supporting_evidence"
ANSWER_STATUS_REALTIME    = "external_information_required"
ANSWER_STATUS_OUT_SCOPE   = "out_of_scope"
ANSWER_STATUS_MODEL_ERROR = "model_unavailable"


class ResearchOrchestrator:
    def __init__(self, llm: OllamaLLM, rag: RAGService) -> None:
        self.llm = llm
        self.rag = rag
        self.planner = PlannerService(llm)
        self.discovery = DiscoveryService()
        self.qa = ResearchQAService(llm)
        self.analysis = AnalysisService(llm)
        self.comparison = ComparisonService(llm)
        self.gaps = GapAnalysisService(llm)
        self.recommendations = RecommendationService(llm)

    async def run(
        self,
        query: str,
        history: list[ChatTurn] | None = None,
        *,
        max_arxiv_results: int = 8,
        force_local_rag: bool | None = None,
        disable_arxiv: bool | None = None,
        preferred_model: str | None = None,
        available_models: list[str] | None = None,
        paper_filter: list[str] | None = None,
    ) -> QueryResponse:
        started = time.perf_counter()
        history_dicts = [turn.model_dump() for turn in (history or [])]
        traces: list[TraceStep] = []
        citations = CitationService()
        papers: list[ArxivPaper] = []
        chunks: list[RetrievedChunk] = []
        answer_status = ANSWER_STATUS_RESEARCH

        skip_arxiv = bool(disable_arxiv) if disable_arxiv is not None else False
        resources_before = snapshot_resources()

        # ------------------------------------------------------------------ #
        # Step 1 — Input guardrail                                            #
        # ------------------------------------------------------------------ #
        guardrail = check_input(query, indexed_chunks=self.rag.indexed_chunks)
        traces.append(
            TraceStep(
                step=1,
                tool="guardrail_input",
                status="ok" if guardrail.passed else "error",
                summary=(
                    f"scope_score={guardrail.scope_score} passed={guardrail.passed}"
                    + (f" flag={guardrail.blocked_reason}" if guardrail.blocked_reason else "")
                ),
                detail=guardrail.model_dump(),
            )
        )

        # Hard-blocked (only truly empty or off-topic-hard queries)
        if not guardrail.passed:
            answer_status = ANSWER_STATUS_OUT_SCOPE
            return self._build_response(
                answer=guardrail.reason,
                answer_status=answer_status,
                query=query,
                plan=heuristic_plan(query),
                guardrail=guardrail,
                orchestration=ModelOrchestrationDecision(
                    selected_model=self.llm.model,
                    rationale="Blocked by input guardrail.",
                ),
                traces=traces,
                citations=[],
                papers=[],
                chunks=[],
                inspector=self._kb_stats(),
                resources_before=resources_before,
                started=started,
            )

        # ------------------------------------------------------------------ #
        # Step 2 — Resolve model BEFORE anything else                        #
        # ------------------------------------------------------------------ #
        # Fix: preferred_model from the UI is ALWAYS honoured.
        # Previous bug: the condition `self.llm.model != OLLAMA_MODEL` meant
        # that when the UI sent the default model name, preferred_model was
        # ignored and the orchestrator never swapped the LLM.
        active_llm = self.llm
        model_rationale = "Using server default LLM."

        if preferred_model and preferred_model.strip():
            target = preferred_model.strip()
            if target != self.llm.model:
                try:
                    active_llm = OllamaLLM(model=target)
                    # Quick reachability check — fail fast before wasting RAG time
                    if not active_llm.ping():
                        raise RuntimeError(f"Ollama is not reachable for model '{target}'.")
                    model_rationale = f"User-selected model: {target}."
                    self._rebind_specialists(active_llm)
                except Exception as exc:
                    err_msg = (
                        f"**Model unavailable:** `{target}` could not be used for this query.\n\n"
                        f"Error: {exc}\n\n"
                        "Please check that the model is installed in Ollama (`ollama list`) and try again."
                    )
                    return self._build_response(
                        answer=err_msg,
                        answer_status=ANSWER_STATUS_MODEL_ERROR,
                        query=query,
                        plan=heuristic_plan(query),
                        guardrail=guardrail,
                        orchestration=ModelOrchestrationDecision(
                            selected_model=target,
                            rationale=f"Model error: {exc}",
                        ),
                        traces=traces,
                        citations=[],
                        papers=[],
                        chunks=[],
                        inspector=self._kb_stats(),
                        resources_before=resources_before,
                        started=started,
                    )
            else:
                model_rationale = f"User-selected model (already active): {target}."
        else:
            # No preference from UI → use complexity routing
            complexity = classify_complexity(query, intent="research_qa")
            selected_model, model_rationale = select_model(
                complexity,
                available_models=available_models,
                preferred_model=None,
            )
            if selected_model and selected_model != self.llm.model:
                try:
                    active_llm = OllamaLLM(model=selected_model)
                    self._rebind_specialists(active_llm)
                except Exception:
                    active_llm = self.llm  # graceful fallback to default

        traces.append(
            TraceStep(
                step=len(traces) + 1,
                tool="model_selection",
                status="ok",
                summary=f"model={active_llm.model}",
                detail={"model": active_llm.model, "rationale": model_rationale},
            )
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Handle realtime queries without touching RAG              #
        # ------------------------------------------------------------------ #
        if guardrail.blocked_reason == "realtime_info_required":
            answer_status = ANSWER_STATUS_REALTIME
            prompt = realtime_fallback_prompt(query)
            answer = await self._llm_generate(active_llm, prompt, traces, label="realtime_fallback")
            return self._build_response(
                answer=answer,
                answer_status=answer_status,
                query=query,
                plan=heuristic_plan(query),
                guardrail=guardrail,
                orchestration=ModelOrchestrationDecision(
                    selected_model=active_llm.model,
                    rationale=model_rationale,
                ),
                traces=traces,
                citations=[],
                papers=[],
                chunks=[],
                inspector=self._kb_stats(),
                resources_before=resources_before,
                started=started,
            )

        # ------------------------------------------------------------------ #
        # Step 4 — Plan                                                       #
        # ------------------------------------------------------------------ #
        plan = await self._timed(
            traces,
            "planner",
            self.planner.plan(query, history_dicts),
            step_override=len(traces) + 1,
            summary_fn=lambda p: (
                f"intent={p.intent.value} tools={p.tools}"
                if isinstance(p, ExecutionPlan) else str(p)
            ),
        )
        if not isinstance(plan, ExecutionPlan):
            plan = heuristic_plan(query)
        if force_local_rag is True:
            plan.requires_local_rag = True
            if "rag_search" not in plan.tools:
                plan.tools = ["rag_search", *plan.tools]

        # ------------------------------------------------------------------ #
        # Step 5 — RAG retrieval                                              #
        # ------------------------------------------------------------------ #
        from llm.prompt_builder import _GENERAL_CONCEPTS as _GC  # noqa: PLC0415
        _q_lower = query.strip().lower()
        _is_likely_conceptual = any(_q_lower.startswith(p) for p in _GC) or len(_q_lower.split()) <= 5
        _is_analysis      = plan.intent == IntentType.ANALYSIS
        _is_gap           = plan.intent == IntentType.GAP_ANALYSIS
        _is_recommendation = plan.intent == IntentType.RECOMMENDATION
        _is_research_qa   = plan.intent == IntentType.RESEARCH_QA
        _is_comparison    = plan.intent == IntentType.COMPARISON
        _use_pdf_only = (
            _is_analysis
            or _is_gap
            or _is_recommendation
            or _is_research_qa
            or _is_comparison
            or _is_likely_conceptual
        )

        # Use larger k for intents that need broad paper coverage
        _retrieval_k = 12 if (_is_analysis or _is_gap or _is_comparison) else (8 if _is_recommendation else None)

        # ── Per-paper isolated retrieval for comparison ───────────────────
        # When paper_filter names ≥2 papers, retrieve each paper's chunks
        # separately so cross-paper contamination is impossible.
        _per_paper_evidence: dict[str, str] | None = None

        if _is_comparison and paper_filter and len(paper_filter) >= 2:
            _per_paper_chunks: dict[str, list] = {}
            _per_paper_cites: dict[str, object] = {}

            for _fname in paper_filter:
                _paper_cites = CitationService()
                _paper_chunks = await asyncio.to_thread(
                    self.rag.search_for_paper, _fname, 20
                )
                for c in _paper_chunks:
                    _paper_cites.register_chunk(c)
                    citations.register_chunk(c)  # also register globally for source rail
                    chunks.append(c)
                _per_paper_chunks[_fname] = _paper_chunks
                _per_paper_cites[_fname] = _paper_cites

                traces.append(
                    TraceStep(
                        step=len(traces) + 1,
                        tool="rag_search",
                        status="ok",
                        summary=f"retrieved {len(_paper_chunks)} chunks for {_fname}",
                        elapsed_ms=0.0,
                        detail={"paper": _fname, "count": len(_paper_chunks)},
                    )
                )

            # Build per-paper evidence blocks (each labelled and capped separately)
            _per_paper_evidence = {}
            _per_paper_char_budget = max(3500, 7000 // len(paper_filter))
            for _fname in paper_filter:
                _ev = _per_paper_cites[_fname].evidence_block(  # type: ignore[union-attr]
                    max_chars=_per_paper_char_budget
                )
                _per_paper_evidence[_fname] = _ev or "(no evidence found for this paper)"

        else:
            # Standard pooled retrieval for non-comparison or single-paper queries
            need_rag = plan.requires_local_rag or "rag_search" in plan.tools
            if need_rag:
                chunks = await self._run_rag(
                    query, traces,
                    pdf_only=_use_pdf_only,
                    k=_retrieval_k,
                    filename_filter=paper_filter or None,
                )
                for chunk in chunks:
                    citations.register_chunk(chunk)

        # ------------------------------------------------------------------ #
        # Step 6 — arXiv discovery                                           #
        # ------------------------------------------------------------------ #
        if (not skip_arxiv) and (
            "arxiv_search" in plan.tools or plan.intent == IntentType.DISCOVERY
        ) and not (_is_comparison and paper_filter):
            # Skip arXiv for explicit paper-filter comparisons —
            # we're comparing local papers, not discovering new ones.
            papers, _ = await self._run_arxiv(plan, max_arxiv_results, traces)
            if not papers and not chunks:
                chunks = await self._run_rag(query, traces, pdf_only=False)
                for chunk in chunks:
                    citations.register_chunk(chunk)
            for paper in papers:
                citations.register_paper(paper)

        evidence = citations.evidence_block()
        has_evidence = bool(
            evidence and evidence.strip() and evidence.strip() != "(no retrieved evidence)"
        )
        if not evidence:
            evidence = "(no retrieved evidence)"

        # ------------------------------------------------------------------ #
        # Step 7 — Query-type classification (after retrieval)               #
        # ------------------------------------------------------------------ #
        retrieved_kinds = [c.source_kind for c in chunks]
        query_type = classify_query_type(
            query,
            evidence if has_evidence else "",
            retrieved_kinds=retrieved_kinds,
        )

        # For a general/conceptual query where retrieved chunks are ALL code,
        # treat it as no-evidence so the LLM answers from training knowledge
        # instead of blending irrelevant code into the answer.
        all_code_chunks = bool(retrieved_kinds) and all(k == "code" for k in retrieved_kinds)
        if all_code_chunks and query_type == "general":
            evidence = "(no retrieved evidence)"
            has_evidence = False

        # ------------------------------------------------------------------ #
        # Step 8 — LLM generation                                            #
        # ------------------------------------------------------------------ #
        _known_specialists = {"research_qa", "analysis", "comparison", "gap_analysis", "recommendation"}

        specialist_names = [name for name in plan.tools if name in _known_specialists]
        if not specialist_names:
            specialist_names = ["research_qa"]
        tool_name = specialist_names[0]

        try:
            if tool_name == "research_qa":
                answer = await self._call_qa(
                    active_llm, query, evidence, history_dicts, query_type, traces
                )
            elif tool_name == "comparison" and _per_paper_evidence:
                # Use per-paper isolated evidence for a genuine comparison
                answer = await self._timed(
                    traces,
                    "comparison",
                    self.comparison.compare(
                        query,
                        evidence,
                        per_paper_evidence=_per_paper_evidence,
                    ),
                    summary_fn=lambda s: bound_text(s, 160),
                )
            else:
                answer = await self._call_specialist(
                    active_llm, tool_name, query, evidence, traces
                )
        except Exception as exc:
            answer = (
                f"**Model error:** `{active_llm.model}` failed to generate a response.\n\n"
                f"Error: {exc}\n\n"
                "Check that Ollama is running and the model is available."
            )
            answer_status = ANSWER_STATUS_MODEL_ERROR
            traces.append(
                TraceStep(
                    step=len(traces) + 1,
                    tool="llm_error",
                    status="error",
                    summary=str(exc)[:200],
                )
            )

        if answer_status != ANSWER_STATUS_MODEL_ERROR:
            if query_type == "realtime":
                answer_status = ANSWER_STATUS_REALTIME
            elif query_type == "out_of_scope":
                answer_status = ANSWER_STATUS_OUT_SCOPE
            elif query_type == "general" and not has_evidence:
                answer_status = ANSWER_STATUS_GENERAL
            elif has_evidence:
                answer_status = ANSWER_STATUS_RESEARCH
            else:
                answer_status = ANSWER_STATUS_NO_EVIDENCE

        if papers:
            listed = "\n".join(
                f"- [{p.title}]({p.url}) (`{p.arxiv_id}`)" for p in papers[:6]
            )
            answer = f"{answer}\n\n### Discovered papers\n{listed}"

        # ------------------------------------------------------------------ #
        # Step 9 — Output grounding validation                               #
        # ------------------------------------------------------------------ #
        retrieved_dicts = [{"text": c.text, "filename": c.filename} for c in chunks]
        grounding_score, grounding_flagged = check_output(
            answer, retrieved_dicts, grounding_threshold=GROUNDING_THRESHOLD
        )
        if grounding_flagged and has_evidence:
            traces.append(
                TraceStep(
                    step=len(traces) + 1,
                    tool="guardrail_output",
                    status="ok",
                    summary=f"grounding_score={grounding_score:.3f} — answer may contain unsupported claims",
                    detail={"grounding_score": grounding_score, "flagged": True},
                )
            )
            if answer_status == ANSWER_STATUS_RESEARCH:
                answer_status = ANSWER_STATUS_PARTIAL

        # ------------------------------------------------------------------ #
        # Assemble response                                                   #
        # ------------------------------------------------------------------ #
        return self._build_response(
            answer=answer,
            answer_status=answer_status,
            query=query,
            plan=plan,
            guardrail=guardrail,
            orchestration=ModelOrchestrationDecision(
                complexity=classify_complexity(query, plan.intent.value),
                selected_model=active_llm.model,
                rationale=model_rationale,
                grounding_score=grounding_score,
                grounding_flagged=grounding_flagged,
            ),
            traces=traces,
            citations=citations.as_list(),
            papers=papers,
            chunks=chunks,
            inspector={
                **(self.rag.last_inspector or {}),
                "llm": dict(getattr(active_llm, "last_call", {}) or {}),
                "resources_before": resources_before,
                "resources_after": snapshot_resources(),
                "kb_stats": self._kb_stats(),
                "lifecycle": {
                    "intent": plan.intent.value,
                    "tools": plan.tools,
                    "arxiv_disabled": skip_arxiv,
                    "query_type": query_type,
                    "selected_model": active_llm.model,
                    "grounding_score": grounding_score,
                    "grounding_flagged": grounding_flagged,
                    "answer_status": answer_status,
                },
            },
            resources_before=resources_before,
            started=started,
        )

    # ---------------------------------------------------------------------- #
    # LLM call helpers                                                        #
    # ---------------------------------------------------------------------- #

    async def _llm_generate(
        self,
        llm: OllamaLLM,
        prompt: str,
        traces: list[TraceStep],
        *,
        label: str = "llm",
        system: str | None = None,
    ) -> str:
        started = time.perf_counter()
        text = await llm.agenerate(prompt, system=system)
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        traces.append(
            TraceStep(
                step=len(traces) + 1,
                tool=label,
                status="ok",
                summary=bound_text(text, 160),
                elapsed_ms=elapsed,
            )
        )
        return text if isinstance(text, str) else str(text)

    async def _call_qa(
        self,
        llm: OllamaLLM,
        query: str,
        evidence: str,
        history_dicts: list[dict],
        query_type: str,
        traces: list[TraceStep],
    ) -> str:
        history_block = format_history(history_dicts)
        prompt = qa_prompt(query, evidence, history_block, query_type=query_type)
        return await self._llm_generate(
            llm,
            prompt,
            traces,
            label="research_qa",
            system="Ground every claim in the evidence when available. Never leave the answer blank.",
        )

    async def _call_specialist(
        self,
        llm: OllamaLLM,
        tool_name: str,
        query: str,
        evidence: str,
        traces: list[TraceStep],
    ) -> str:
        """Call analysis/comparison/gap/recommendation with correct service."""
        # Rebind in case the LLM changed since construction
        self._rebind_specialists(llm)
        started = time.perf_counter()
        if tool_name == "analysis":
            coro = self.analysis.analyze(query, evidence)
        elif tool_name == "comparison":
            coro = self.comparison.compare(query, evidence)
        elif tool_name == "gap_analysis":
            coro = self.gaps.find_gaps(query, evidence)
        elif tool_name == "recommendation":
            coro = self.recommendations.recommend(query, evidence)
        else:
            from llm.prompt_builder import qa_prompt
            coro = llm.agenerate(qa_prompt(query, evidence, "(none)", query_type="research"))
        result = await coro
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        text = result if isinstance(result, str) else str(result)
        traces.append(
            TraceStep(
                step=len(traces) + 1,
                tool=tool_name,
                status="ok",
                summary=bound_text(text, 160),
                elapsed_ms=elapsed,
            )
        )
        return text

    # ---------------------------------------------------------------------- #
    # KB stats helper                                                         #
    # ---------------------------------------------------------------------- #

    def _kb_stats(self) -> dict[str, Any]:
        """Return knowledge-base level statistics for the RAG trace panel."""
        metas = getattr(self.rag.store, "metadatas", None) or []
        total_chunks = self.rag.indexed_chunks
        pdf_chunks: dict[str, int] = {}
        pdf_pages: dict[str, set[int]] = {}
        code_chunks = 0
        for meta in metas:
            fname = str((meta or {}).get("filename") or "")
            kind = str((meta or {}).get("source_kind") or "")
            page = (meta or {}).get("page")
            if kind == "code":
                code_chunks += 1
            else:
                pdf_chunks[fname] = pdf_chunks.get(fname, 0) + 1
                if page:
                    pdf_pages.setdefault(fname, set()).add(int(page))

        sources = []
        for pdf_path in self.rag.list_pdfs():
            fname = pdf_path.name
            # Try relative path match too (metadata stores relative path)
            chunks_for = pdf_chunks.get(fname, 0)
            if chunks_for == 0:
                # try relative-path key
                for k, v in pdf_chunks.items():
                    if k.endswith(fname):
                        chunks_for = v
                        break
            pages_for = len(pdf_pages.get(fname, set()))
            sources.append({
                "filename": fname,
                "chunks": chunks_for,
                "pages_extracted": pages_for,
                "source_kind": "local_pdf",
            })

        from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
        from vector_db.embeddings import last_query_embed_stats
        embed_stats = last_query_embed_stats()

        return {
            "total_chunks": total_chunks,
            "pdf_count": len(self.rag.list_pdfs()),
            "code_files_indexed": len(self.rag.list_code_files()) if total_chunks > 0 else 0,
            "code_chunks": code_chunks,
            "sources": sources,
            "chunking": {
                "chunk_size_tokens": CHUNK_SIZE,
                "overlap_tokens": CHUNK_OVERLAP,
                "step_tokens": max(1, CHUNK_SIZE - CHUNK_OVERLAP),
                "method": "whitespace_token_sliding_window",
            },
            "embedding": {
                "model": EMBEDDING_MODEL,
                "dimension": embed_stats.get("dimension"),
            },
            "vector_store": self.rag.backend_name,
        }

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _rebind_specialists(self, llm: OllamaLLM) -> None:
        self.qa = ResearchQAService(llm)
        self.analysis = AnalysisService(llm)
        self.comparison = ComparisonService(llm)
        self.gaps = GapAnalysisService(llm)
        self.recommendations = RecommendationService(llm)

    def _build_response(
        self,
        *,
        answer: str,
        answer_status: str,
        query: str,
        plan: ExecutionPlan,
        guardrail: GuardrailResult,
        orchestration: ModelOrchestrationDecision,
        traces: list[TraceStep],
        citations: list,
        papers: list,
        chunks: list,
        inspector: dict[str, Any],
        resources_before: dict[str, Any],
        started: float,
    ) -> QueryResponse:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        llm_call = inspector.get("llm") or {}
        pres = build_presentation(
            query=query,
            answer=answer,
            intent=plan.intent,
            citations=citations,
            papers=papers,
            chunks=chunks,
            indexed_docs=len(self.rag.list_pdfs()) or self.rag.indexed_chunks,
        )
        pres["answer_status"] = answer_status
        return QueryResponse(
            answer=answer,
            intent=plan.intent,
            plan=plan,
            citations=citations,
            traces=traces,
            papers=papers,
            chunks_used=len(chunks),
            inspector=inspector,
            guardrail=guardrail,
            orchestration=orchestration,
            telemetry={
                "elapsed_ms": elapsed,
                "vector_backend": self.rag.backend_name,
                "indexed_chunks": self.rag.indexed_chunks,
                "dense_search": getattr(self.rag.store, "last_telemetry", {}),
                "pdf_count": len(self.rag.list_pdfs()),
                "code_files": len(self.rag.list_code_files()),
                "llm": llm_call,
                "answer_status": answer_status,
                "model": orchestration.selected_model,
            },
            presentation=pres,
        )

    async def _run_rag(self, query: str, traces: list[TraceStep],
                       pdf_only: bool = False, k: int | None = None,
                       filename_filter: list[str] | None = None) -> list[RetrievedChunk]:
        started = time.perf_counter()
        try:
            if self.rag.indexed_chunks == 0:
                await asyncio.to_thread(self.rag.ingest_all, rebuild=False)
            chunks = await asyncio.to_thread(
                self.rag.search, query, k, pdf_only, filename_filter
            )
            mode = []
            if pdf_only:
                mode.append("PDF-only")
            if filename_filter:
                mode.append(f"filter={','.join(filename_filter)}")
            if k:
                mode.append(f"k={k}")
            mode_str = " (" + ", ".join(mode) + ")" if mode else ""
            traces.append(
                TraceStep(
                    step=len(traces) + 1,
                    tool="rag_search",
                    status="ok",
                    summary=f"retrieved {len(chunks)} hybrid-ranked chunks{mode_str}",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    detail={"count": len(chunks), "pdf_only": pdf_only,
                            "filename_filter": filename_filter,
                            **getattr(self.rag.store, "last_telemetry", {})},
                )
            )
            return chunks
        except Exception as exc:  # noqa: BLE001
            traces.append(
                TraceStep(
                    step=len(traces) + 1,
                    tool="rag_search",
                    status="error",
                    summary=str(exc),
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )
            return []

    async def _run_arxiv(
        self,
        plan: ExecutionPlan,
        max_results: int,
        traces: list[TraceStep],
    ) -> tuple[list[ArxivPaper], dict]:
        started = time.perf_counter()
        papers, telemetry = await asyncio.to_thread(
            self.discovery.search, plan.arxiv_query or "", max_results=max_results
        )
        status = "ok" if telemetry.get("ok") else "error"
        traces.append(
            TraceStep(
                step=len(traces) + 1,
                tool="arxiv_search",
                status=status,
                summary=f"found {len(papers)} papers" if papers else telemetry.get("error", "no results"),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=telemetry,
            )
        )
        return papers, telemetry

    async def _timed(
        self,
        traces: list[TraceStep],
        tool: str,
        coro: Any,
        *,
        step_override: int | None = None,
        summary_fn=None,
    ) -> Any:
        started = time.perf_counter()
        result = await coro
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        summary = summary_fn(result) if summary_fn else str(result)[:160]
        traces.append(
            TraceStep(
                step=step_override if step_override is not None else len(traces) + 1,
                tool=tool,
                status="ok",
                summary=summary,
                elapsed_ms=elapsed,
            )
        )
        return result
