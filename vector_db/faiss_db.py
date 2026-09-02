"""FAISS L2 vector store with persistence and bounded-k search telemetry."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray

from config import FAISS_INDEX_PATH, FAISS_META_PATH, INDEX_DIR, RETRIEVAL_K
from vector_db.embeddings import embed_query, embed_texts, last_query_embed_stats

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(
        self,
        index_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> None:
        self.index_path = index_path or FAISS_INDEX_PATH
        self.meta_path = meta_path or FAISS_META_PATH
        self.index: faiss.IndexFlatL2 | None = None
        self.metadatas: list[dict[str, Any]] = []
        self.documents: list[str] = []
        self.ids: list[str] = []
        self.last_telemetry: dict[str, Any] = {}
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.load()

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.metadatas = payload.get("metadatas", [])
            self.documents = payload.get("documents", [])
            self.ids = payload.get("ids", [])
            logger.info("Loaded FAISS index with %s vectors", self.ntotal)
            return
        self.index = None
        self.metadatas = []
        self.documents = []
        self.ids = []

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(
                {
                    "ids": self.ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.index = None
        self.metadatas = []
        self.documents = []
        self.ids = []
        for path in (self.index_path, self.meta_path):
            if path.exists():
                path.unlink()

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not documents:
            return 0
        vectors = embed_texts(documents, normalize=False)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.save()
        return len(documents)

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = k if k is not None else RETRIEVAL_K
        if self.index is None or self.ntotal == 0:
            self.last_telemetry = {
                "backend": "faiss",
                "ntotal": 0,
                "k_requested": requested_k,
                "k_used": 0,
                "elapsed_ms": 0.0,
            }
            return []

        k_used = max(1, min(requested_k, self.ntotal))
        query_vec: NDArray[np.float32] = embed_query(query, normalize=False).reshape(1, -1)
        search_started = time.perf_counter()
        distances, indices = self.index.search(query_vec, k_used)
        search_ms = (time.perf_counter() - search_started) * 1000
        hits: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self.documents):
                continue
            hits.append(
                {
                    "id": self.ids[idx],
                    "text": self.documents[idx],
                    "score": float(dist),
                    "distance": float(dist),
                    "metadata": self.metadatas[idx],
                }
            )
        embed_stats = last_query_embed_stats()
        self.last_telemetry = {
            "backend": "faiss",
            "metric": "L2",
            "ntotal": self.ntotal,
            "k_requested": requested_k,
            "k_used": k_used,
            "returned": len(hits),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "faiss_search_ms": round(search_ms, 3),
            "query_embed_ms": embed_stats.get("latency_ms"),
            "embedding_dimension": embed_stats.get("dimension"),
            "embedding_model": embed_stats.get("model"),
        }
        return hits
