"""Three-model evaluation runner with resume support.

Persistence
-----------
After every single question the partial results are flushed to
eval/results/partial_{model}.json so that a laptop sleep / crash / restart
loses at most one question.  On the next run, already-completed questions are
loaded from the partial file and skipped — only the remaining ones are run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.schemas import ChatTurn
from config import EVAL_DISABLE_ARXIV, EVAL_MODELS, EVAL_RESULTS_DIR, ROOT_DIR, ensure_data_dirs
from eval.code_tests import run_code_test
from eval.metrics import score_item
from llm.llm_model import OllamaLLM
from services.orchestrator import ResearchOrchestrator
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
DATASET_PATH = ROOT_DIR / "eval" / "dataset.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset() -> dict[str, Any]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def list_local_models(llm: OllamaLLM) -> set[str]:
    try:
        listing = llm.client.list()
        models = getattr(listing, "models", None) or listing.get("models", [])  # type: ignore[union-attr]
        names: set[str] = set()
        for item in models:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name:
                names.add(str(name))
                names.add(str(name).split(":")[0])
        return names
    except Exception:
        return set()


def _retrieved_from_inspector(inspector: dict[str, Any]) -> list[dict[str, Any]]:
    return list(inspector.get("retrieved_chunks") or [])


def _partial_path(model: str) -> Path:
    """Path to the per-model partial results file."""
    safe = model.replace(":", "_").replace("/", "_")
    return EVAL_RESULTS_DIR / f"partial_{safe}.json"


def _load_partial(model: str, limit: int | None) -> dict[str, Any]:
    """Load any previously saved partial results for this model.

    Returns a dict mapping question id → completed row.
    Clears the partial file if the saved item_count differs from the
    current limit (run configuration changed).
    """
    path = _partial_path(model)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        saved_limit = data.get("item_count")
        if limit is not None and saved_limit != limit:
            # Different limit → old partial is stale, start fresh
            path.unlink(missing_ok=True)
            return {}
        rows = data.get("rows") or []
        return {r["id"]: r for r in rows}
    except Exception:
        return {}


def _save_partial(model: str, rows: list[dict[str, Any]], item_count: int) -> None:
    """Flush completed rows to disk after every question."""
    try:
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _partial_path(model).write_text(
            json.dumps({"model": model, "item_count": item_count, "rows": rows},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save partial results for %s: %s", model, exc)


def _clear_partials(models: list[str]) -> None:
    for model in models:
        _partial_path(model).unlink(missing_ok=True)


# ── Single-item runner ────────────────────────────────────────────────────────

async def run_one(
    orchestrator: ResearchOrchestrator,
    item: dict[str, Any],
    *,
    disable_arxiv: bool,
) -> dict[str, Any]:
    response = await orchestrator.run(
        item["question"],
        history=[],
        force_local_rag=True,
        disable_arxiv=disable_arxiv,
    )
    inspector = response.inspector or {}
    retrieved = _retrieved_from_inspector(inspector)
    llm_meta = inspector.get("llm") or {}
    code = run_code_test(item.get("code_test"), response.answer)
    scores = score_item(item, response.answer, retrieved, code)
    elapsed = (response.telemetry or {}).get("elapsed_ms")
    prompt_tokens = llm_meta.get("prompt_tokens") or 0
    completion_tokens = llm_meta.get("completion_tokens") or 0
    if not prompt_tokens:
        raw = llm_meta.get("raw_prompt") or ""
        prompt_tokens = max(1, len(raw) // 4)
        completion_tokens = max(1, len(response.answer) // 4)
    return {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "answer": response.answer,
        "intent": response.intent.value,
        "retrieved_context": retrieved,
        "raw_prompt": llm_meta.get("raw_prompt"),
        "metrics": scores,
        "latency_ms": elapsed,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
            "tokens_per_sec": llm_meta.get("tokens_per_sec"),
        },
        "resources": inspector.get("resources_after"),
        "rag_trace": {
            "question": item["question"],
            "retrieved_context": retrieved,
            "llm_response": response.answer,
            "retrieval_label": scores["retrieval_label"],
            "hallucinated_despite_context": scores["hallucinated_despite_context"],
            "answer_correct_enough": scores["answer_correct_enough"],
        },
    }


# ── Summarise ────────────────────────────────────────────────────────────────

def summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    n = len(rows)
    code_rows = [r for r in rows if (r.get("metrics") or {}).get("code_test", {}).get("applicable")]
    code_pass = sum(1 for r in code_rows if r["metrics"]["code_test"].get("passed"))
    rss = [((r.get("resources") or {}).get("rss_mb") or 0) for r in rows]
    cpu = [((r.get("resources") or {}).get("process_cpu_percent") or 0) for r in rows]
    gpu_used = [
        ((r.get("resources") or {}).get("gpu") or {}).get("memory_used_mb")
        for r in rows
        if ((r.get("resources") or {}).get("gpu") or {}).get("available")
    ]
    return {
        "n": n,
        "correctness_mean": round(sum(r["metrics"]["correctness"] for r in rows) / n, 4),
        "relevance_mean": round(sum(r["metrics"]["relevance"] for r in rows) / n, 4),
        "retrieval_quality_mean": round(sum(r["metrics"]["retrieval_quality"] for r in rows) / n, 4),
        "hallucination_rate_mean": round(sum(r["metrics"]["hallucination_rate"] for r in rows) / n, 4),
        "test_pass_rate": (round(code_pass / len(code_rows), 4) if code_rows else None),
        "code_items": len(code_rows),
        "latency_ms_mean": round(sum((r.get("latency_ms") or 0) for r in rows) / n, 2),
        "tokens_mean": round(sum((r.get("tokens") or {}).get("total") or 0 for r in rows) / n, 2),
        "tokens_per_sec_mean": round(
            sum((r.get("tokens") or {}).get("tokens_per_sec") or 0 for r in rows) / n, 3,
        ),
        "rss_mb_mean": round(sum(rss) / n, 2),
        "cpu_percent_mean": round(sum(cpu) / n, 2),
        "gpu_memory_used_mb_mean": round(sum(gpu_used) / len(gpu_used), 2) if gpu_used else None,
    }


# ── Main evaluation runner ────────────────────────────────────────────────────

async def run_evaluation(
    models: list[str] | None = None,
    *,
    limit: int | None = None,
    rag: RAGService | None = None,
    disable_arxiv: bool | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    ensure_data_dirs()
    dataset = load_dataset()
    items = dataset["items"]
    if limit:
        items = items[:limit]
    skip_arxiv = EVAL_DISABLE_ARXIV if disable_arxiv is None else disable_arxiv
    rag = rag or RAGService()
    if rag.indexed_chunks == 0:
        rag.ingest_all(rebuild=False)

    probe = OllamaLLM()
    installed = list_local_models(probe)
    selected = models or list(EVAL_MODELS)
    by_model: dict[str, Any] = {}
    skipped: list[str] = []

    total_items = len(items)
    active_models = [
        m for m in selected
        if not (installed and m not in installed
                and m.split(":")[0] not in installed
                and not any(str(x).startswith(m) or str(x).startswith(m.split(":")[0]) for x in installed))
    ]
    grand_total = total_items * max(1, len(active_models))
    completed_so_far = 0

    for model in selected:
        base = model.split(":")[0]
        if installed and model not in installed and base not in installed and not any(
            str(m).startswith(model) or str(m).startswith(base) for m in installed
        ):
            skipped.append(model)
            by_model[model] = {"error": "model_not_installed", "installed_hint": sorted(installed)[:12]}
            continue

        # Load any partial results saved before a previous sleep/crash
        cached: dict[str, Any] = _load_partial(model, total_items)
        if cached:
            logger.info("Resuming %s: %d/%d questions already done", model, len(cached), total_items)

        llm = OllamaLLM(model=model)
        orchestrator = ResearchOrchestrator(llm, rag)

        # Restore previously completed rows in original order
        rows: list[dict[str, Any]] = []
        for item in items:
            if item["id"] in cached:
                rows.append(cached[item["id"]])

        for idx, item in enumerate(items, start=1):
            # Skip already-completed questions (resume support)
            if item["id"] in cached:
                completed_so_far += 1
                if progress_cb:
                    progress_cb({
                        "phase": "running (resumed)",
                        "model": model,
                        "item_id": item.get("id"),
                        "done": completed_so_far,
                        "total": grand_total,
                        "current_model_index": selected.index(model) if model in selected else 0,
                        "items_per_model": total_items,
                        "resumed": True,
                    })
                continue

            if progress_cb:
                progress_cb({
                    "phase": "running",
                    "model": model,
                    "item_id": item.get("id"),
                    "done": completed_so_far,
                    "total": grand_total,
                    "current_model_index": selected.index(model) if model in selected else 0,
                    "items_per_model": total_items,
                    "resumed": False,
                })

            row = await run_one(orchestrator, item, disable_arxiv=skip_arxiv)
            row["model"] = model
            # Insert in original order
            rows_by_id = {r["id"]: r for r in rows}
            rows_by_id[row["id"]] = row
            rows = [rows_by_id.get(i["id"]) for i in items if i["id"] in rows_by_id]

            # ── Persist after EVERY question so sleep/crash loses nothing ──
            _save_partial(model, rows, total_items)

            completed_so_far += 1

        by_model[model] = {"summary": summarize_model(rows), "items": rows}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_requested": selected,
        "models_skipped": skipped,
        "item_count": len(items),
        "constants": dataset.get("hold_constant"),
        "metric_definitions": Path(__file__).with_name("metrics.py").read_text(encoding="utf-8")[:2500],
        "disable_arxiv": skip_arxiv,
        "by_model": by_model,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    from eval.analyze import write_analysis
    write_analysis(payload)

    # Clean up partial files now that we have a complete result
    _clear_partials(selected)
    return payload


# ── Custom single-question evaluator ─────────────────────────────────────────

async def run_custom_question(
    question: str,
    models: list[str],
    *,
    rag: RAGService | None = None,
    disable_arxiv: bool = True,
) -> list[dict[str, Any]]:
    """Run a single user-supplied question against each model and return scored rows."""
    rag = rag or RAGService()
    if rag.indexed_chunks == 0:
        rag.ingest_all(rebuild=False)

    # Build a synthetic eval item with no gold keywords (score is answer quality proxy)
    item = {
        "id": "CUSTOM",
        "category": "custom",
        "question": question,
        "gold_keywords": question.lower().split()[:8],  # use question words as proxy keywords
        "expected_files": [],
        "must_cite": False,
    }

    results: list[dict[str, Any]] = []
    for model in models:
        llm = OllamaLLM(model=model)
        orchestrator = ResearchOrchestrator(llm, rag)
        try:
            row = await run_one(orchestrator, item, disable_arxiv=disable_arxiv)
            row["model"] = model
        except Exception as exc:  # noqa: BLE001
            row = {
                "id": "CUSTOM",
                "category": "custom",
                "question": question,
                "model": model,
                "answer": f"Error: {exc}",
                "metrics": {"correctness": 0, "relevance": 0, "retrieval_quality": 0,
                            "hallucination_rate": 1, "code_test": {"applicable": False}},
                "latency_ms": 0,
                "tokens": {"total": 0},
            }
        results.append(row)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run 3-model RAG evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", type=str, default=None, help="comma-separated Ollama tags")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    asyncio.run(run_evaluation(models=models, limit=args.limit))
    print(f"Wrote {EVAL_RESULTS_DIR / 'latest.json'}")


if __name__ == "__main__":
    main()
