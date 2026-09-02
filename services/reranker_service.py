"""Hybrid dense + BM25 search with Reciprocal Rank Fusion re-ranking."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from rank_bm25 import BM25Okapi

from config import RETRIEVAL_K, RERANK_TOP_K, RRF_K

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


class HybridReranker:
    """Combines dense vector ranks with BM25 via RRF.

    A cross-encoder can be enabled later without changing callers; the default
    path stays CPU-light for 7B-class local setups.
    """

    def __init__(self, use_cross_encoder: bool = False) -> None:
        self.use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

    def _maybe_cross_encoder(self):
        if not self.use_cross_encoder:
            return None
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cross-encoder unavailable, using RRF only: %s", exc)
                self._cross_encoder = False
        return self._cross_encoder or None

    def fuse(
        self,
        query: str,
        dense_hits: list[dict[str, Any]],
        corpus_docs: list[str] | None = None,
        corpus_meta: list[dict[str, Any]] | None = None,
        corpus_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = top_k or RERANK_TOP_K
        bm25_hits: list[dict[str, Any]] = []
        if corpus_docs:
            tokenized = [tokenize(doc) for doc in corpus_docs]
            if any(tokenized):
                bm25 = BM25Okapi(tokenized)
                scores = bm25.get_scores(tokenize(query))
                ranked = sorted(
                    enumerate(scores),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[: max(RETRIEVAL_K, limit)]
                for rank, (idx, score) in enumerate(ranked, start=1):
                    if score <= 0:
                        continue
                    meta = (corpus_meta or [{}] * len(corpus_docs))[idx]
                    doc_id = (corpus_ids or [str(i) for i in range(len(corpus_docs))])[idx]
                    bm25_hits.append(
                        {
                            "id": doc_id,
                            "text": corpus_docs[idx],
                            "score": float(score),
                            "rank": rank,
                            "metadata": meta,
                            "origin": "bm25",
                        }
                    )

        fused: dict[str, dict[str, Any]] = {}
        for origin, hits in (("dense", dense_hits), ("bm25", bm25_hits)):
            for rank, hit in enumerate(hits, start=1):
                key = str(hit.get("id") or f"{origin}-{rank}")
                current = fused.setdefault(
                    key,
                    {
                        "id": key,
                        "text": hit.get("text", ""),
                        "metadata": hit.get("metadata") or {},
                        "rrf": 0.0,
                        "origins": [],
                    },
                )
                if not current["text"]:
                    current["text"] = hit.get("text", "")
                if not current["metadata"]:
                    current["metadata"] = hit.get("metadata") or {}
                current["rrf"] += rrf_score(rank)
                current["origins"].append(origin)

        ranked_fused = sorted(fused.values(), key=lambda item: item["rrf"], reverse=True)
        encoder = self._maybe_cross_encoder()
        if encoder is not None and ranked_fused:
            pairs = [(query, item["text"]) for item in ranked_fused[: max(limit * 2, 8)]]
            try:
                ce_scores = encoder.predict(pairs)
                for item, score in zip(ranked_fused, ce_scores, strict=False):
                    item["rrf"] = float(item["rrf"]) + 0.15 * (1.0 / (1.0 + math.exp(-float(score))))
                ranked_fused.sort(key=lambda item: item["rrf"], reverse=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cross-encoder scoring failed: %s", exc)

        return ranked_fused[:limit]
