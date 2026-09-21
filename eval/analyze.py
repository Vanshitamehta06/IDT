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
# Category interpretation engine (rule-based, no LLM call)
# ---------------------------------------------------------------------------

# Human-readable descriptions for each category
_CAT_DESCRIPTIONS: dict[str, str] = {
    "code_explanation":        "Understanding what a specific function or module does",
    "code_retrieval":          "Finding which file or component handles a specific task",
    "dependency_understanding":"Identifying which components depend on or use a service",
    "bug_analysis":            "Diagnosing what could cause incorrect or unexpected behaviour",
    "code_generation":         "Writing new code (functions, unit tests, utilities)",
    "refactoring":             "Suggesting concrete improvements to existing code",
    "rag_qa":                  "Answering questions grounded in the indexed research papers",
    "analysis":                "Deep structural analysis of a paper (problem, method, contributions)",
    "comparison":              "Side-by-side comparison of methods or papers",
    "gap_analysis":            "Identifying open research problems and future directions",
    "recommendation":          "Recommending next papers or research directions",
    "discovery":               "Finding related literature from external sources",
    "repo_multi_file":         "Questions spanning multiple source files in the codebase",
    "repo_retrieval":          "Specific retrieval of implementation details from the codebase",
    "hallucination_probe":     "Testing whether the model fabricates facts not in the evidence",
    "refactoring":             "Suggesting concrete code improvements",
}

# Why a model might score high or low on each category — templates filled with actual data
_HIGH_CORRECT = "{model} scores high ({val:.2f}) on {cat} — it correctly identifies the relevant content and uses the right terminology."
_LOW_CORRECT  = "{model} scores low ({val:.2f}) on {cat} — it struggles to locate or reproduce the expected answer keywords."
_HIGH_HALL    = "{model} has a high hallucination rate ({val:.2f}) on {cat} — many answer sentences lack overlap with retrieved context, suggesting the model is drawing on training data rather than evidence."
_LOW_HALL     = "{model} has a low hallucination rate ({val:.2f}) on {cat} — answers are well-grounded in retrieved content."
_HIGH_LAT     = "{model} is slow ({val:.0f} ms) on {cat} — likely due to long retrieved context or complex generation."
_LOW_LAT      = "{model} is fast ({val:.0f} ms) on {cat}."

_CAT_REASONS: dict[str, dict[str, str]] = {
    "code_explanation": {
        "high_correct":  "The model correctly describes what the function does by finding and quoting the right code chunk.",
        "low_correct":   "The model may miss the relevant function entirely, or describe it at too high a level without the expected keywords.",
        "high_hall":     "The model is likely filling in plausible-sounding behaviour not actually in the retrieved code.",
        "low_retrieval": "The code chunk for this function was not retrieved — the model answered from training knowledge alone.",
    },
    "code_retrieval": {
        "high_correct":  "The model correctly identifies the file/component by matching the retrieval query to the right chunk.",
        "low_correct":   "The expected filename is not surfacing in the answer — either retrieval missed it or the model ignored the citation.",
        "high_hall":     "The model is naming files or components that are not in the retrieved evidence.",
        "low_retrieval": "The relevant file chunk was not in the top-k retrieved results.",
    },
    "dependency_understanding": {
        "high_correct":  "The model traces the dependency chain correctly from the retrieved code.",
        "low_correct":   "Dependency relationships span multiple files; if retrieval only returns one file the model sees an incomplete picture.",
        "high_hall":     "The model invents dependency relationships not supported by the retrieved chunks.",
        "low_retrieval": "Multi-file dependency questions require chunks from several files simultaneously — a known limitation of single-pass retrieval.",
    },
    "bug_analysis": {
        "high_correct":  "The model identifies the likely failure mode by reasoning over the retrieved code and error context.",
        "low_correct":   "Bug analysis requires understanding subtle code paths; the model may produce a plausible but incorrect diagnosis.",
        "high_hall":     "The model is speculating about bugs not evidenced in the retrieved code.",
        "low_retrieval": "The relevant code section was not retrieved, so the model cannot ground its analysis.",
    },
    "code_generation": {
        "high_correct":  "The model generates syntactically valid code that passes the unit test checker.",
        "low_correct":   "The generated code either has a syntax error, wrong function name, or incorrect logic.",
        "high_hall":     "Generated code contains statements or imports not grounded in the retrieved context.",
        "low_retrieval": "Code generation is less dependent on retrieval — performance reflects model training strength.",
    },
    "refactoring": {
        "high_correct":  "The model proposes a concrete, evidence-backed improvement using the right terminology.",
        "low_correct":   "The suggestion is too generic or does not reference the specific method/pattern asked about.",
        "high_hall":     "The model is proposing changes not supported by the retrieved source code.",
        "low_retrieval": "The target method chunk was not retrieved so the model cannot make a specific suggestion.",
    },
    "rag_qa": {
        "high_correct":  "The model answers correctly using the retrieved research paper chunks.",
        "low_correct":   "The answer misses expected keywords — either retrieval brought the wrong chunks or the model did not cite them.",
        "high_hall":     "The model is generating claims not supported by the retrieved paper content.",
        "low_retrieval": "The relevant paper section was not in the top retrieved chunks — a retrieval quality issue.",
    },
}

_DEFAULT_REASONS: dict[str, str] = {
    "high_correct":  "The model correctly addresses the question using the retrieved content.",
    "low_correct":   "The model's answer misses expected keywords — retrieval may have returned insufficient or irrelevant chunks.",
    "high_hall":     "Many sentences in the model's answer lack grounding in retrieved context.",
    "low_retrieval": "The relevant content was not retrieved for this category.",
}


def _interpret_model_on_category(
    model: str,
    cat: str,
    metrics: dict[str, float],
    all_models_cat: dict[str, dict[str, float]],
) -> str:
    """Generate a plain-English interpretation for one model on one category."""
    corr  = metrics.get("correctness", 0)
    hall  = metrics.get("hallucination_rate", 0)
    ret   = metrics.get("retrieval_quality", 0)
    lat   = metrics.get("latency_ms", 0)
    n     = int(metrics.get("n", 0))
    reasons = _CAT_REASONS.get(cat, _DEFAULT_REASONS)

    # Is this model above or below the cross-model average for this category?
    other_corr = [v.get("correctness", 0) for m, v in all_models_cat.items() if m != model]
    avg_corr = sum(other_corr) / len(other_corr) if other_corr else corr

    lines: list[str] = []

    # Correctness judgement
    if corr >= 0.7:
        lines.append(f"**Correctness ({corr:.2f}) — Strong.** {reasons['high_correct']}")
    elif corr >= 0.4:
        delta = corr - avg_corr
        trend = "above" if delta > 0.05 else ("below" if delta < -0.05 else "on par with")
        lines.append(
            f"**Correctness ({corr:.2f}) — Moderate** ({trend} the other models at {avg_corr:.2f}). "
            f"{reasons['low_correct']}"
        )
    else:
        lines.append(f"**Correctness ({corr:.2f}) — Weak.** {reasons['low_correct']}")

    # Hallucination judgement
    if hall >= 0.6:
        lines.append(f"**Hallucination ({hall:.2f}) — High ⚠.** {reasons['high_hall']}")
    elif hall <= 0.2:
        lines.append(f"**Hallucination ({hall:.2f}) — Low ✓.** The model stays grounded in evidence.")
    else:
        lines.append(f"**Hallucination ({hall:.2f}) — Moderate.** Some sentences lack retrieval grounding.")

    # Retrieval quality
    if ret < 0.3:
        lines.append(f"**Retrieval quality ({ret:.2f}) — Poor.** {reasons['low_retrieval']}")
    elif ret >= 0.7:
        lines.append(f"**Retrieval quality ({ret:.2f}) — Good.** Relevant content was retrieved.")

    # Latency note (only if notably slow)
    if lat > 60000:
        lines.append(f"**Latency ({lat:.0f} ms) — Slow.** Generation took significantly longer than average for this category.")

    # n warning
    if n == 1:
        lines.append(f"*Note: only 1 question in this category — interpret with caution.*")

    return "\n\n".join(lines)


def _build_category_interpretations(
    category_full_breakdown: dict[str, dict[str, dict[str, float]]],
    summaries: dict[str, dict],
) -> dict[str, dict[str, str]]:
    """Build {category: {model: interpretation_text}} for the UI.

    Purely rule-based — no LLM call, instant, works offline.
    """
    # Collect all categories
    all_cats: set[str] = set()
    for model_cats in category_full_breakdown.values():
        all_cats.update(model_cats.keys())

    interpretations: dict[str, dict[str, str]] = {}

    for cat in sorted(all_cats):
        cat_interps: dict[str, str] = {}
        # Per-category metrics for all models (for relative comparison)
        all_models_cat = {
            m: category_full_breakdown[m].get(cat, {})
            for m in category_full_breakdown
        }
        for model, model_cats in category_full_breakdown.items():
            if cat not in model_cats:
                continue
            metrics = model_cats[cat]
            cat_interps[model] = _interpret_model_on_category(
                model, cat, metrics, all_models_cat
            )

        # Add an overall category summary comparing models
        if len(cat_interps) >= 2:
            scores = {
                m: category_full_breakdown[m].get(cat, {}).get("correctness", 0)
                for m in category_full_breakdown if cat in category_full_breakdown[m]
            }
            best_m  = max(scores, key=lambda x: scores[x]) if scores else None
            worst_m = min(scores, key=lambda x: scores[x]) if scores else None
            cat_desc = _CAT_DESCRIPTIONS.get(cat, cat.replace("_", " ").title())
            gap = round((scores.get(best_m, 0) - scores.get(worst_m, 0)), 3) if best_m and worst_m else 0
            overall = (
                f"**Category: {cat_desc}**\n\n"
                f"Best performing model: **{best_m}** ({scores.get(best_m, 0):.3f} correctness). "
                f"Weakest: **{worst_m}** ({scores.get(worst_m, 0):.3f}). "
                f"Performance gap: {gap:.3f}.\n\n"
            )
            if gap < 0.1:
                overall += "All models perform similarly on this category."
            elif gap > 0.4:
                overall += "There is a large performance gap — this category strongly differentiates the models."
            else:
                overall += "Moderate variation across models on this category."
            cat_interps["_overall"] = overall

        interpretations[cat] = cat_interps

    return interpretations


# ---------------------------------------------------------------------------
# Chart data builder
# ---------------------------------------------------------------------------

def _build_chart_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Build chart-ready data structures consumed directly by the Streamlit UI."""
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
        scatter.append({
            "model": model,
            "latency_ms": bar_latency[model],
            "correctness": bar_correctness[model],
            "hallucination": bar_hallucination[model],
            "retrieval": bar_retrieval[model],
        })

    max_latency = max(bar_latency.values(), default=1) or 1
    radar: dict[str, dict[str, float]] = {}
    for model, s in summaries.items():
        lat_norm = round(1.0 - min(bar_latency.get(model, 0) / max_latency, 1.0), 4)
        radar[model] = {
            "Correctness": bar_correctness.get(model, 0),
            "Relevance": round(s.get("relevance_mean") or 0.0, 4),
            "Retrieval": bar_retrieval.get(model, 0),
            "Grounding": round(1.0 - bar_hallucination.get(model, 0), 4),
            "Speed": lat_norm,
        }

    # Per-category breakdown for ALL metrics
    category_full_breakdown: dict[str, dict[str, dict[str, float]]] = {}
    category_breakdown: dict[str, dict[str, float]] = {}

    for model, block in by_model.items():
        if not isinstance(block, dict):
            continue
        items = block.get("items") or []
        cat_data: dict[str, dict[str, list[float]]] = {}
        for row in items:
            cat = str(row.get("category") or "other")
            m_scores = row.get("metrics") or {}
            lat = row.get("latency_ms") or 0.0
            tok = (row.get("tokens") or {}).get("total") or 0
            if cat not in cat_data:
                cat_data[cat] = {
                    "correctness": [], "relevance": [], "retrieval_quality": [],
                    "hallucination_rate": [], "latency_ms": [], "tokens": [],
                }
            cat_data[cat]["correctness"].append(m_scores.get("correctness") or 0.0)
            cat_data[cat]["relevance"].append(m_scores.get("relevance") or 0.0)
            cat_data[cat]["retrieval_quality"].append(m_scores.get("retrieval_quality") or 0.0)
            cat_data[cat]["hallucination_rate"].append(m_scores.get("hallucination_rate") or 0.0)
            cat_data[cat]["latency_ms"].append(lat)
            cat_data[cat]["tokens"].append(tok)

        cat_summary: dict[str, dict[str, float]] = {}
        cat_correctness_only: dict[str, float] = {}
        for cat, metrics_lists in cat_data.items():
            cat_summary[cat] = {
                k: round(sum(v) / len(v), 4) if v else 0.0
                for k, v in metrics_lists.items()
            }
            cat_summary[cat]["n"] = len(cat_data[cat]["correctness"])
            cat_correctness_only[cat] = cat_summary[cat]["correctness"]
        category_full_breakdown[model] = cat_summary
        category_breakdown[model] = cat_correctness_only

    # ── Category interpretations (rule-based, no LLM) ───────────────────────
    category_interpretations = _build_category_interpretations(
        category_full_breakdown, summaries
    )

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
        "category_full_breakdown": category_full_breakdown,
        "category_interpretations": category_interpretations,
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
        "## Category-wise metric breakdown",
        "",
        "Each row is the mean across all questions in that category.",
        "Columns: Correctness | Relevance | Retrieval | Hallucination ↓ | Latency ms ↓ | n (questions)",
        "",
    ]

    # Gather all categories across all models
    all_cats: list[str] = []
    for model, block in (payload.get("by_model") or {}).items():
        if not isinstance(block, dict):
            continue
        for row in (block.get("items") or []):
            cat = str(row.get("category") or "other")
            if cat not in all_cats:
                all_cats.append(cat)

    # Build a table per model
    for model, block in (payload.get("by_model") or {}).items():
        if not isinstance(block, dict) or not block.get("items"):
            continue
        lines += [f"### {model}", ""]
        lines.append("| Category | Correct | Relevant | Retrieval | Hallucin ↓ | Latency ms | n |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        # Compute per-category stats inline
        cat_data: dict[str, dict[str, list]] = {}
        for row in (block.get("items") or []):
            cat = str(row.get("category") or "other")
            m = row.get("metrics") or {}
            if cat not in cat_data:
                cat_data[cat] = {"c": [], "r": [], "rq": [], "h": [], "l": []}
            cat_data[cat]["c"].append(m.get("correctness") or 0.0)
            cat_data[cat]["r"].append(m.get("relevance") or 0.0)
            cat_data[cat]["rq"].append(m.get("retrieval_quality") or 0.0)
            cat_data[cat]["h"].append(m.get("hallucination_rate") or 0.0)
            cat_data[cat]["l"].append(row.get("latency_ms") or 0.0)

        def _mean(lst: list) -> str:
            return f"{sum(lst)/len(lst):.3f}" if lst else "—"

        for cat in all_cats:
            if cat not in cat_data:
                continue
            d = cat_data[cat]
            n = len(d["c"])
            lines.append(
                f"| {cat} | {_mean(d['c'])} | {_mean(d['r'])} | {_mean(d['rq'])} "
                f"| {_mean(d['h'])} | {_mean(d['l'])} | {n} |"
            )
        lines.append("")

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
