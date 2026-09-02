"""Pydantic contracts for requests, responses, traces, and citations."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class GuardrailResult(BaseModel):
    passed: bool
    reason: str = ""
    scope_score: float = 0.0  # 0–1 relevance to research domain
    blocked_reason: Literal["out_of_scope", "no_sources", "rate_limited", ""] = ""


# ---------------------------------------------------------------------------
# Model orchestration
# ---------------------------------------------------------------------------

class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class ModelOrchestrationDecision(BaseModel):
    complexity: ComplexityLevel = ComplexityLevel.SIMPLE
    selected_model: str = ""
    rationale: str = ""
    grounding_score: float = 1.0   # 0–1; <threshold means potentially unsupported
    grounding_flagged: bool = False


# ---------------------------------------------------------------------------
# Prompt suggestions
# ---------------------------------------------------------------------------

class SuggestRequest(BaseModel):
    partial: str = Field(..., min_length=1, max_length=500)
    context_titles: list[str] = Field(default_factory=list)  # indexed PDF names for context

    @field_validator("partial")
    @classmethod
    def strip_partial(cls, v: str) -> str:
        return v.strip()


class SuggestResponse(BaseModel):
    suggestions: list[str]
    partial: str


class IntentType(str, Enum):
    RESEARCH_QA = "research_qa"
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    GAP_ANALYSIS = "gap_analysis"
    RECOMMENDATION = "recommendation"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)
    max_arxiv_results: int = Field(default=8, ge=1, le=25)
    force_local_rag: bool | None = None
    vector_backend: Literal["faiss", "chroma"] | None = None
    model: str | None = None
    disable_arxiv: bool | None = None
    # Optional list of PDF filenames (basename, e.g. "rag_survey.pdf") to
    # restrict retrieval to specific documents.  Used by Compare Papers.
    paper_filter: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class EvalRunRequest(BaseModel):
    models: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=30)
    disable_arxiv: bool = True


class IngestRequest(BaseModel):
    rebuild: bool = False
    vector_backend: Literal["faiss", "chroma"] | None = None


class ExecutionPlan(BaseModel):
    intent: IntentType = IntentType.RESEARCH_QA
    tools: list[str] = Field(default_factory=lambda: ["rag_search", "research_qa"])
    arxiv_query: str = ""
    requires_local_rag: bool = True
    comparison_targets: list[str] = Field(default_factory=list)
    rationale: str = ""


class Citation(BaseModel):
    source_id: int
    tag: str
    kind: Literal["local_pdf", "arxiv"] = "local_pdf"
    title: str
    locator: str = ""
    snippet: str = ""
    url: str | None = None
    filename: str | None = None
    page: int | None = None
    section: str | None = None
    arxiv_id: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = 0.0
    filename: str = ""
    page: int | None = None
    section: str | None = None
    source_kind: Literal["local_pdf", "arxiv", "code"] = "local_pdf"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published: str = ""
    updated: str = ""
    url: str = ""
    categories: list[str] = Field(default_factory=list)


class TraceStep(BaseModel):
    step: int
    tool: str
    status: Literal["ok", "skipped", "error", "fallback"]
    summary: str
    elapsed_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    intent: IntentType
    plan: ExecutionPlan
    citations: list[Citation] = Field(default_factory=list)
    traces: list[TraceStep] = Field(default_factory=list)
    papers: list[ArxivPaper] = Field(default_factory=list)
    chunks_used: int = 0
    inspector: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    presentation: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    guardrail: GuardrailResult = Field(default_factory=lambda: GuardrailResult(passed=True))
    orchestration: ModelOrchestrationDecision = Field(default_factory=ModelOrchestrationDecision)


class ModelsResponse(BaseModel):
    models: list[str]
    default_model: str
    eval_models: list[str]


class EvalChartData(BaseModel):
    """Pre-computed chart payloads for the UI (no JS charting library needed)."""
    radar: dict[str, Any] = Field(default_factory=dict)   # {model: {metric: value}}
    bar_latency: dict[str, float] = Field(default_factory=dict)    # model -> latency_ms
    bar_correctness: dict[str, float] = Field(default_factory=dict)
    bar_hallucination: dict[str, float] = Field(default_factory=dict)
    bar_retrieval: dict[str, float] = Field(default_factory=dict)
    scatter: list[dict[str, Any]] = Field(default_factory=list)    # [{model, latency, correctness}]
    category_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)  # model -> category -> correctness


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    ollama_host: str
    ollama_model: str
    ollama_reachable: bool
    vector_backend: str
    indexed_chunks: int
    pdf_count: int
    embedding_model: str
    detail: str = ""


class IngestResponse(BaseModel):
    status: str
    files_ingested: list[str]
    chunks_indexed: int
    backend: str
    elapsed_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None
    chunk_step_tokens: int | None = None
