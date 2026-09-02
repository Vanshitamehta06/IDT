"""Write a quantitative analysis report and chart-data JSON from evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import EVAL_RESULTS_DIR, ROOT_DIR


def _best(summaries: dict[str, dict], key: str, higher: bool = True) -> str | None:
    ranked = [(m, s.get(key)) for m, s in summaries.items() if s.get(key) is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[1], reverse=higher)
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Chart data builder
# ---------------------------------------------------------------------------

def _build_chart_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Build chart-ready data structures consumed directly by the Streamlit UI.

    Returns a dict with keys:
      radar            – {model: {metric: value}}  (5 metrics, normalised 0-1, latency inverted)
      bar_correctness  – {model: float}
      bar_hallucination– {model: float}
      bar_retrieval    – {model: float}
      bar_latency      – {model: float}  (raw ms)
      scatter          – [{model, latency_ms, correctness, hallucination}]
      category_breakdown – {model: {category: correctness_mean}}
      tokens_per_sec   – {model: float}
      rss_mb           – {model: float}
    """
    by_model: dict[str, Any] = payload.get("by_model") or {}
    summaries: dict[str, dict] = {
        m: b["summary"]
        for m, b in by_model.items()
        if isinstance(b, dict) and b.get("summary")
    }

    bar_correctness: dict[str, float] = {}
    bar_hallucination: dict[str, float] = {}
    bar_retrieval: dict[str, float] = {}
    bar_latency: dict[str, float] = {}
    tokens_per_sec: dict[str, float] = {}
    rss_mb: dict[str, float] = {}
    scatter: list[dict[str, Any]] = []

    for model, s in summaries.items():
        bar_correctness[model] = round(s.get("correctness_mean") or 0.0, 4)
        bar_hallucination[model] = round(s.get("hallucination_rate_mean") or 0.0, 4)
        bar_retrieval[model] = round(s.get("retrieval_quality_mean") or 0.0, 4)
        bar_latency[model] = round(s.get("latency_ms_mean") or 0.0, 2)
        tokens_per_sec[model] = round(s.get("tokens_per_sec_mean") or 0.0, 3)
        rss_mb[model] = round(s.get("rss_mb_mean") or 0.0, 2)
        scatter.append(
            {
                "model": model,
                "latency_ms": bar_latency[model],
                "correctness": bar_correctness[model],
                "hallucination": bar_hallucination[model],
                "retrieval": bar_retrieval[model],
            }
        )

    # Radar: normalise all metrics to 0-1 (latency is inverted — lower is better)
    max_latency = max(bar_latency.values(), default=1) or 1
    radar: dict[str, dict[str, float]] = {}
    for model, s in summaries.items():
        lat_norm = round(1.0 - min(bar_latency.get(model, 0) / max_latency, 1.0), 4)
        radar[model] = {
            "Correctness": bar_correctness.get(model, 0),
            "Relevance": round(s.get("relevance_mean") or 0.0, 4),
            "Retrieval": bar_retrieval.get(model, 0),
            "Grounding": round(1.0 - bar_hallucination.get(model, 0), 4),  # inverted hallucination
            "Speed": lat_norm,
        }

    # Per-category correctness breakdown
    category_breakdown: dict[str, dict[str, float]] = {}
    for model, block in by_model.items():
        if not isinstance(block, dict):
            continue
        items = block.get("items") or []
        cat_scores: dict[str, list[float]] = {}
        for row in items:
            cat = str(row.get("category") or "other")
            score = (row.get("metrics") or {}).get("correctness") or 0.0
            cat_scores.setdefault(cat, []).append(score)
        category_breakdown[model] = {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in cat_scores.items()
        }

    return {
        "radar": radar,
        "bar_correctness": bar_correctness,
        "bar_hallucination": bar_hallucination,
        "bar_retrieval": bar_retrieval,
        "bar_latency": bar_latency,
        "tokens_per_sec": tokens_per_sec,
        "rss_mb": rss_mb,
        "scatter": scatter,
        "category_breakdown": category_breakdown,
    }


# ---------------------------------------------------------------------------
# Markdown analysis writer
# ---------------------------------------------------------------------------

def write_analysis(payload: dict[str, Any]) -> Path:
    summaries: dict[str, dict] = {}
    for model, block in (payload.get("by_model") or {}).items():
        if isinstance(block, dict) and block.get("summary"):
            summaries[model] = block["summary"]

    lines = [
        "# Three-model evaluation analysis",
        "",
        f"Generated: {payload.get('generated_at')}",
        f"Items per model: {payload.get('item_count')}",
        f"Models skipped (not installed): {', '.join(payload.get('models_skipped') or []) or 'none'}",
        "",
        "## Constants (held fixed)",
        "",
    ]
    for c in payload.get("constants") or []:
        lines.append(f"- {c}")

    lines += [
        "",
        "## Metric definitions",
        "",
        "See `eval/metrics.py` for exact formulas. Summary:",
        "- **Correctness**: gold-keyword coverage (citation penalty if required).",
        "- **Relevance**: Jaccard overlap of question vs answer tokens.",
        "- **Retrieval quality**: expected-file hit rate, else keyword coverage of chunks.",
        "- **Hallucination rate**: share of sentences poorly overlapping retrieved context.",
        "- **Test-pass rate**: fraction of codegen items whose extracted Python passes unit checks.",
        "- **Latency**: end-to-end `orchestrator.run` milliseconds.",
        "- **Tokens**: Ollama prompt+completion counts (or char/4 estimate).",
        "- **Resources**: process RSS MB, CPU %, optional nvidia-smi GPU memory.",
        "",
        "## Quantitative summary",
        "",
        "| Model | Correctness | Relevance | Retrieval | Hallucination ↓ | Test-pass | Latency ms ↓ | Tokens | tok/s | RSS MB | CPU % | GPU MB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, s in summaries.items():
        lines.append(
            "| {model} | {correctness_mean} | {relevance_mean} | {retrieval_quality_mean}"
            " | {hallucination_rate_mean} | {test_pass_rate} | {latency_ms_mean}"
            " | {tokens_mean} | {tokens_per_sec_mean} | {rss_mb_mean}"
            " | {cpu_percent_mean} | {gpu_memory_used_mb_mean} |".format(
                model=model,
                **{
                    k: s.get(k)
                    for k in [
                        "correctness_mean",
                        "relevance_mean",
                        "retrieval_quality_mean",
                        "hallucination_rate_mean",
                        "test_pass_rate",
                        "latency_ms_mean",
                        "tokens_mean",
                        "tokens_per_sec_mean",
                        "rss_mb_mean",
                        "cpu_percent_mean",
                        "gpu_memory_used_mb_mean",
                    ]
                },
            )
        )

    acc = _best(summaries, "correctness_mean", True)
    hall = _best(summaries, "hallucination_rate_mean", False)
    ret = _best(summaries, "retrieval_quality_mean", True)
    test = _best(summaries, "test_pass_rate", True)
    lat = _best(summaries, "latency_ms_mean", False)
    rss = _best(summaries, "rss_mb_mean", False)

    lines += [
        "",
        "## Model comparison — interpretation",
        "",
        f"- Highest correctness: **{acc}**",
        f"- Lowest hallucination rate: **{hall}**",
        f"- Best retrieval-grounded file/keyword hits: **{ret}**",
        f"- Highest code test-pass rate: **{test}**",
        f"- Lowest mean latency: **{lat}**",
        f"- Lowest mean RSS: **{rss}**",
        "",
    ]
    if acc and lat and acc != lat:
        lines.append(
            f"- Quality–latency trade-off: **{acc}** leads accuracy while **{lat}** is faster. "
            "The most accurate model is not automatically the most efficient."
        )
    elif acc:
        lines.append(f"- **{acc}** is both the accuracy leader and (tied) latency leader on this run.")

    lines += [
        "",
        "## RAG trace analysis — retrieval → context → answer",
        "",
        "Selected traces (first model with items):",
        "",
    ]
    for model, block in (payload.get("by_model") or {}).items():
        rows = block.get("items") if isinstance(block, dict) else None
        if not rows:
            continue
        picks = []
        for label in (
            "relevant_information_retrieved",
            "irrelevant_information_retrieved",
            "important_information_missed",
        ):
            found = next((r for r in rows if r.get("metrics", {}).get("retrieval_label") == label), None)
            if found:
                picks.append(found)
        halluc = next((r for r in rows if r.get("metrics", {}).get("hallucinated_despite_context")), None)
        if halluc:
            picks.append(halluc)
        for row in picks[:6]:
            ctx = row.get("retrieved_context") or []
            preview = (ctx[0].get("text") if ctx else "")[:400]
            lines += [
                f"### {row.get('id')} ({model}) — {row.get('metrics', {}).get('retrieval_label')}",
                f"**Question:** {row.get('question')}",
                f"**Retrieved (first chunk):** {preview}",
                f"**Answer excerpt:** {(row.get('answer') or '')[:400]}",
                f"**Correct enough:** {row.get('metrics', {}).get('answer_correct_enough')}; "
                f"**Hallucinated despite context:** {row.get('metrics', {}).get('hallucinated_despite_context')}",
                "",
            ]
        break

    lines += [
        "## Repository-level understanding",
        "",
        "Questions Q09–Q13, Q17–Q19, Q25 ask about *this* codebase across multiple files.",
        "If retrieval quality is high but correctness is low, the LLM failed to compose a multi-file story.",
        "If retrieval quality is low, chunking/embeddings missed the relevant modules.",
        "",
    ]

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_RESULTS_DIR / "analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    report = ROOT_DIR / "eval" / "ANALYSIS.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Write chart data JSON alongside the markdown                        #
    # ------------------------------------------------------------------ #
    charts = _build_chart_data(payload)
    charts_path = EVAL_RESULTS_DIR / "charts.json"
    charts_path.write_text(json.dumps(charts, ensure_ascii=False, indent=2), encoding="utf-8")

    return path
