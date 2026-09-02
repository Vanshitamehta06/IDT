"""Shared embedding encoder (sentence-transformers, CPU-friendly)."""

from __future__ import annotations

import logging
from functools import lru_cache

import time

import numpy as np
from numpy.typing import NDArray

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_last_query_embed: dict = {
    "latency_ms": 0.0,
    "dimension": 0,
    "model": EMBEDDING_MODEL,
}


@lru_cache(maxsize=1)
def get_encoder():
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def embedding_dimension() -> int:
    encoder = get_encoder()
    dim = getattr(encoder, "get_sentence_embedding_dimension", lambda: None)()
    if dim:
        return int(dim)
    probe = embed_texts(["dimension probe"], normalize=False)
    return int(probe.shape[1]) if probe.size else 0


def last_query_embed_stats() -> dict:
    return dict(_last_query_embed)


def embed_texts(texts: list[str], *, normalize: bool = True) -> NDArray[np.float32]:
    if not texts:
        dim = 384
        try:
            dim = embedding_dimension()
        except Exception:
            pass
        return np.zeros((0, dim), dtype=np.float32)
    encoder = get_encoder()
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str, *, normalize: bool = True) -> NDArray[np.float32]:
    started = time.perf_counter()
    vector = embed_texts([text], normalize=normalize)[0]
    _last_query_embed.update(
        {
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "dimension": int(vector.shape[0]),
            "model": EMBEDDING_MODEL,
            "query_chars": len(text),
        }
    )
    return vector
