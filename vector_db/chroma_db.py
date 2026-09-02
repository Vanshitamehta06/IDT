"""ChromaDB wrapper using get_or_create_collection and UUID chunk ids."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_K
from vector_db.embeddings import embed_query, embed_texts, last_query_embed_stats

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(self, collection_name: str | None = None) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name or COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"},
        )
        self.last_telemetry: dict[str, Any] = {}

    @property
    def ntotal(self) -> int:
        return int(self.collection.count())

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            logger.debug("Chroma collection delete skipped")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def add(
        self,
        ids: list[str] | None,
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        if not documents:
            return 0
        chunk_ids = ids or [str(uuid.uuid4()) for _ in documents]
        if len(chunk_ids) != len(documents):
            chunk_ids = [str(uuid.uuid4()) for _ in documents]
        embeddings = embed_texts(documents, normalize=False).tolist()
        self.collection.add(
            ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(documents)

    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        started = time.perf_counter()
        requested_k = k if k is not None else RETRIEVAL_K
        count = self.ntotal
        if count == 0:
            self.last_telemetry = {
                "backend": "chroma",
                "ntotal": 0,
                "k_requested": requested_k,
                "k_used": 0,
                "elapsed_ms": 0.0,
            }
            return []

        k_used = max(1, min(requested_k, count))
        query_embedding = embed_query(query, normalize=False).tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k_used,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        hits: list[dict[str, Any]] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            hits.append(
                {
                    "id": chunk_id,
                    "text": text or "",
                    "score": float(dist) if dist is not None else 0.0,
                    "distance": float(dist) if dist is not None else 0.0,
                    "metadata": meta or {},
                }
            )
        embed_stats = last_query_embed_stats()
        self.last_telemetry = {
            "backend": "chroma",
            "metric": "L2",
            "ntotal": count,
            "k_requested": requested_k,
            "k_used": k_used,
            "returned": len(hits),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_embed_ms": embed_stats.get("latency_ms"),
            "embedding_dimension": embed_stats.get("dimension"),
            "embedding_model": embed_stats.get("model"),
        }
        return hits
