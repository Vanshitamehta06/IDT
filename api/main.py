"""FastAPI server — lifespan-managed engines, rate limiting, and all endpoints."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    EvalRunRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
    SuggestRequest,
    SuggestResponse,
)
from config import (
    EMBEDDING_MODEL,
    EVAL_MODELS,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    VECTOR_BACKEND,
    ensure_data_dirs,
)
from llm.llm_model import OllamaLLM
from services.orchestrator import ResearchOrchestrator
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

runtime: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Simple in-memory sliding-window rate limiter
# ---------------------------------------------------------------------------
# Maps client IP → list of request timestamps in the current window
_rate_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    if not RATE_LIMIT_ENABLED:
        return True
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    history = _rate_store[client_ip]
    # Drop timestamps outside the window
    _rate_store[client_ip] = [t for t in history if t >= window_start]
    if len(_rate_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_store[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    llm = OllamaLLM()
    rag = RAGService(backend=VECTOR_BACKEND)
    try:
        rag.ingest_all(rebuild=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup ingest deferred: %s", exc)
    runtime["llm"] = llm
    runtime["rag"] = rag
    runtime["orchestrator"] = ResearchOrchestrator(llm, rag)
    yield
    runtime.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Multi-Agent Academic Research Assistant",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rate-limit middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate-limit the expensive endpoints
    limited_paths = {"/query", "/ingest", "/eval/run"}
    if request.url.path in limited_paths:
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests "
                        f"per {RATE_LIMIT_WINDOW_SECONDS}s window."
                    )
                },
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def _engines() -> tuple[OllamaLLM, RAGService, ResearchOrchestrator]:
    try:
        return runtime["llm"], runtime["rag"], runtime["orchestrator"]
    except KeyError as exc:
        raise HTTPException(status_code=503, detail="Engines are still starting") from exc


def _indexed_names(rag: RAGService) -> set[str]:
    metas = getattr(rag.store, "metadatas", None) or []
    names: set[str] = set()
    for meta in metas:
        name = str((meta or {}).get("filename") or "")
        if name.endswith(".pdf"):
            names.add(Path(name).name)
            names.add(name)
    return names


def _list_installed_models(llm: OllamaLLM) -> list[str]:
    """Return list of model tags installed in Ollama."""
    try:
        listing = llm.client.list()
        models = getattr(listing, "models", None) or listing.get("models", [])  # type: ignore[union-attr]
        names: list[str] = []
        seen: set[str] = set()
        for item in models:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name and name not in seen:
                names.append(str(name))
                seen.add(str(name))
        return sorted(names)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    llm, rag, _ = _engines()
    reachable = llm.ping()
    indexed = rag.indexed_chunks
    pdfs = len(rag.list_pdfs())
    status = "ok" if reachable else "degraded"
    if indexed == 0 and pdfs == 0:
        status = "degraded"
    return HealthResponse(
        status=status,
        ollama_host=OLLAMA_HOST,
        ollama_model=OLLAMA_MODEL,
        ollama_reachable=reachable,
        vector_backend=rag.backend_name,
        indexed_chunks=indexed,
        pdf_count=pdfs,
        embedding_model=EMBEDDING_MODEL,
        detail="Ready" if reachable else "Ollama is not reachable; answers will be limited.",
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    llm, _, _ = _engines()
    installed = _list_installed_models(llm)
    return ModelsResponse(
        models=installed,
        default_model=OLLAMA_MODEL,
        eval_models=EVAL_MODELS,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    _, rag, _ = _engines()
    if payload.vector_backend and payload.vector_backend != rag.backend_name:
        rag_switched = RAGService(backend=payload.vector_backend)
        runtime["rag"] = rag_switched
        runtime["orchestrator"] = ResearchOrchestrator(runtime["llm"], rag_switched)
        rag = rag_switched
    result = rag.ingest_all(rebuild=payload.rebuild)
    return IngestResponse(**result)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    _, rag, orchestrator = _engines()
    if payload.vector_backend and payload.vector_backend != rag.backend_name:
        rag = RAGService(backend=payload.vector_backend)
        runtime["rag"] = rag
        orchestrator = ResearchOrchestrator(runtime["llm"], rag)
        runtime["orchestrator"] = orchestrator

    # Resolve available models for intelligent routing
    llm_for_listing = runtime["llm"]
    available = _list_installed_models(llm_for_listing) or EVAL_MODELS

    # If the caller explicitly requests a model, swap the orchestrator's LLM
    if payload.model and payload.model != orchestrator.llm.model:
        llm = OllamaLLM(model=payload.model)
        orchestrator = ResearchOrchestrator(llm, rag)
        runtime["query_orchestrator"] = orchestrator

    return await orchestrator.run(
        payload.query,
        payload.history,
        max_arxiv_results=payload.max_arxiv_results,
        force_local_rag=payload.force_local_rag,
        disable_arxiv=payload.disable_arxiv,
        preferred_model=payload.model,
        available_models=available,
        paper_filter=payload.paper_filter or None,
    )


# ---------------------------------------------------------------------------
# Prompt suggestions
# ---------------------------------------------------------------------------

@app.post("/suggest", response_model=SuggestResponse)
def suggest(payload: SuggestRequest) -> SuggestResponse:
    """Return 4–5 contextual research question suggestions for a partial query."""
    from services.suggest_service import suggest as _suggest

    _, rag, _ = _engines()
    # Pass indexed PDF filenames as domain context
    pdf_names = [p.name for p in rag.list_pdfs()]
    suggestions = _suggest(
        partial=payload.partial,
        context_titles=payload.context_titles or pdf_names,
        n=5,
    )
    return SuggestResponse(suggestions=suggestions, partial=payload.partial)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@app.get("/library")
def library() -> dict:
    from services.library_service import list_library

    _, rag, _ = _engines()
    items = list_library(indexed_filenames=_indexed_names(rag))
    return {"count": len(items), "papers": items, "indexed_chunks": rag.indexed_chunks}


@app.post("/library/upload")
async def library_upload(file: UploadFile = File(...)) -> dict:
    from services.library_service import save_pdf

    _, rag, orchestrator = _engines()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        path = save_pdf(filename=filename, data=data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

    # Rebuild index so the new paper is immediately searchable across all documents.
    # This ensures retrieval always searches ALL indexed research papers, not just
    # whichever paper was ingested first.
    try:
        ingest_result = rag.ingest_all(rebuild=True)
        logger.info("Auto-rebuild after upload: %s", ingest_result.get("chunks_indexed"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-rebuild after upload failed: %s", exc)
        ingest_result = {"status": "rebuild_failed", "error": str(exc)}

    return {
        "saved": path.name,
        "size_bytes": len(data),
        "index_status": ingest_result.get("status"),
        "chunks_indexed": ingest_result.get("chunks_indexed"),
    }


@app.delete("/library/{filename}")
def library_delete(filename: str) -> dict:
    from services.library_service import delete_pdf

    _, rag, _ = _engines()
    deleted = delete_pdf(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"{filename} not found in library")
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@app.get("/eval/dataset")
def eval_dataset() -> dict:
    from eval.runner import load_dataset

    return load_dataset()


@app.get("/eval/status")
def eval_status() -> dict:
    return runtime.get("eval_status") or {"running": False}


@app.get("/eval/results")
def eval_results() -> dict:
    from config import EVAL_RESULTS_DIR

    latest = EVAL_RESULTS_DIR / "latest.json"
    analysis = EVAL_RESULTS_DIR / "analysis.md"
    charts = EVAL_RESULTS_DIR / "charts.json"
    if not latest.exists():
        return {"available": False}
    payload = json.loads(latest.read_text(encoding="utf-8"))
    payload["available"] = True
    payload["analysis_markdown"] = analysis.read_text(encoding="utf-8") if analysis.exists() else ""
    payload["charts"] = json.loads(charts.read_text(encoding="utf-8")) if charts.exists() else {}
    return payload


@app.get("/eval/results/model/{model_name}")
def eval_results_for_model(model_name: str) -> dict:
    """Return per-item detail for a specific model from the latest eval run."""
    from config import EVAL_RESULTS_DIR

    latest = EVAL_RESULTS_DIR / "latest.json"
    if not latest.exists():
        raise HTTPException(status_code=404, detail="No evaluation results available.")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    by_model = payload.get("by_model") or {}
    # Support partial match (e.g. "llama3" matches "llama3:8b")
    matched_key = None
    for key in by_model:
        if model_name in key or key.startswith(model_name.split(":")[0]):
            matched_key = key
            break
    if matched_key is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(by_model.keys())}",
        )
    return {"model": matched_key, **by_model[matched_key]}


@app.post("/eval/run")
async def eval_run(payload: EvalRunRequest) -> dict:
    import asyncio

    task = runtime.get("eval_task")
    if task is not None and not task.done():
        raise HTTPException(status_code=409, detail="An evaluation run is already in progress")

    async def _job() -> None:
        from eval.runner import run_evaluation

        runtime["eval_status"] = {
            "running": True,
            "phase": "starting",
            "models": payload.models or EVAL_MODELS,
        }
        try:
            result = await run_evaluation(
                models=payload.models,
                limit=payload.limit,
                rag=runtime.get("rag"),
                disable_arxiv=payload.disable_arxiv,
                progress_cb=lambda status: runtime.__setitem__("eval_status", {**status, "running": True}),
            )
            runtime["eval_status"] = {
                "running": False,
                "finished": True,
                "item_count": result.get("item_count"),
                "models_skipped": result.get("models_skipped"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Evaluation failed")
            runtime["eval_status"] = {"running": False, "finished": False, "error": str(exc)}

    runtime["eval_task"] = asyncio.create_task(_job())
    return {"status": "started", "models": payload.models or EVAL_MODELS, "limit": payload.limit}
