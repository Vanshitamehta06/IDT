"""Three-model evaluation runner. Same app, prompts, questions, and knowledge base."""

from __future__ import annotations

import asyncio
import json
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

DATASET_PATH = ROOT_DIR / "eval" / "dataset.json"


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
            sum((r.get("tokens") or {}).get("tokens_per_sec") or 0 for r in rows) / n,
            3,
        ),
        "rss_mb_mean": round(sum(rss) / n, 2),
        "cpu_percent_mean": round(sum(cpu) / n, 2),
        "gpu_memory_used_mb_mean": round(sum(gpu_used) / len(gpu_used), 2) if gpu_used else None,
    }


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

    for model in selected:
        base = model.split(":")[0]
        if installed and model not in installed and base not in installed and not any(
            str(m).startswith(model) or str(m).startswith(base) for m in installed
        ):
            skipped.append(model)
            by_model[model] = {"error": "model_not_installed", "installed_hint": sorted(installed)[:12]}
            continue
        llm = OllamaLLM(model=model)
        orchestrator = ResearchOrchestrator(llm, rag)
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(items, start=1):
            if progress_cb:
                progress_cb(
                    {
                        "phase": "running",
                        "model": model,
                        "item_id": item.get("id"),
                        "done": idx - 1,
                        "total": len(items) * max(1, len(selected) - len(skipped)),
                    }
                )
            row = await run_one(orchestrator, item, disable_arxiv=skip_arxiv)
            row["model"] = model
            rows.append(row)
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
    return payload


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
