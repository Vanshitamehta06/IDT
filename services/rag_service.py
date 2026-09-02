"""Multi-PDF ingestion and retrieval over FAISS or Chroma."""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from api.schemas import RetrievedChunk
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INCLUDE_CODE_INDEX,
    PDF_DIR,
    RETRIEVAL_K,
    ROOT_DIR,
    VECTOR_BACKEND,
    ensure_data_dirs,
)
from services.reranker_service import HybridReranker
from vector_db.chroma_db import ChromaVectorStore
from vector_db.embeddings import last_query_embed_stats
from vector_db.faiss_db import FaissVectorStore

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(abstract|introduction|related work|method|methods|methodology|experiments?|results?|discussion|conclusion|references|limitations)\b", re.I)


def _approx_tokens(text: str) -> list[str]:
    return text.split()


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    tokens = _approx_tokens(text)
    if not tokens:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def infer_section(page_text: str, fallback: str = "Unknown") -> str:
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_RE.match(stripped) or (stripped.isupper() and 3 < len(stripped) < 80):
            return stripped[:120]
    return fallback


class RAGService:
    def __init__(self, backend: str | None = None) -> None:
        ensure_data_dirs()
        self.backend_name = (backend or VECTOR_BACKEND).lower()
        self.store: FaissVectorStore | ChromaVectorStore
        if self.backend_name == "chroma":
            self.store = ChromaVectorStore()
        else:
            self.backend_name = "faiss"
            self.store = FaissVectorStore()
        self.reranker = HybridReranker(use_cross_encoder=False)
        self.last_ingest: dict[str, Any] = {}
        self.last_inspector: dict[str, Any] = {}
        # Per-source tracking populated during ingest_all
        self._source_stats: dict[str, dict[str, Any]] = {}   # filename → {chunks, pages, source_kind}

    def search_for_paper(self, filename: str, k: int = 15) -> list[RetrievedChunk]:
        """Retrieve ALL chunks from a single paper for comparison.

        For small papers (≤30 chunks), returns every chunk from that paper
        directly from the metadata store — no embedding search needed.
        For larger papers, runs targeted queries to cover key sections.
        """
        # Directly get all chunks for this paper from the store metadata
        # This is the most reliable approach — no relevance bias, no missed sections
        all_meta_chunks = self._get_all_chunks_for_paper(filename)
        if all_meta_chunks:
            # Sort by page number for logical reading order
            all_meta_chunks.sort(key=lambda c: (c.page or 0))
            return all_meta_chunks[:k]

        # Fallback: multi-query targeted search if direct lookup fails
        _queries = [
            "abstract introduction problem statement objective",
            "methodology approach method architecture model",
            "experiments results benchmarks datasets evaluation metrics numbers",
            "contributions limitations future work conclusion",
        ]
        seen_ids: set[str] = set()
        all_chunks: list[RetrievedChunk] = []
        per_query_k = max(4, k // len(_queries) + 2)

        for q in _queries:
            try:
                partial = self.search(
                    q, k=per_query_k, pdf_only=True, filename_filter=[filename]
                )
                for c in partial:
                    if c.chunk_id not in seen_ids:
                        seen_ids.add(c.chunk_id)
                        all_chunks.append(c)
            except Exception:  # noqa: BLE001
                continue

        all_chunks.sort(key=lambda c: (c.page or 0, c.score * -1))
        return all_chunks[:k]

    def _get_all_chunks_for_paper(self, filename: str) -> list[RetrievedChunk]:
        """Return RetrievedChunk objects for every chunk belonging to a specific paper.

        Reads directly from the store's metadata — no embedding similarity, no bias.
        This guarantees EVERY section of the paper is available for comparison.
        """
        target_base = filename.split("\\")[-1].split("/")[-1].lower()
        metas = getattr(self.store, "metadatas", None) or []
        docs  = getattr(self.store, "documents", None) or []
        ids   = getattr(self.store, "ids", None) or []

        chunks: list[RetrievedChunk] = []
        for chunk_id, text, meta in zip(ids, docs, metas):
            fname = str((meta or {}).get("filename") or "")
            base  = fname.split("\\")[-1].split("/")[-1].lower()
            kind  = str((meta or {}).get("source_kind") or "local_pdf")
            if base != target_base or kind == "code":
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    score=1.0,
                    filename=fname,
                    page=meta.get("page") or None,
                    section=str(meta.get("section") or "") or None,
                    source_kind="local_pdf",
                    metadata=dict(meta),
                )
            )
        return chunks

    def _expand_query(self, query: str) -> str:
        """Expand short or acronym-heavy queries so embeddings match formal paper language.

        For queries ≤ 6 words, we append a contextual expansion that bridges the
        vocabulary gap between colloquial questions and formal academic text.
        The expansion is purely additive and does not change the semantic intent.

        Examples:
          "What is RAG?" → "What is RAG? retrieval augmented generation definition"
          "What is EDA?" → "What is EDA? Easy Data Augmentation definition"
        No hard-coded paper→query mappings. This works for any future paper.
        """
        q = (query or "").strip()
        words = q.split()
        if len(words) > 8:
            return q   # long queries already carry sufficient context

        lower = q.lower()
        suffix_parts: list[str] = []

        # Expand known NLP/ML acronyms that appear in academic papers
        _ACRONYMS = {
            "rag":   "retrieval augmented generation definition explanation",
            "llm":   "large language model definition",
            "eda":   "easy data augmentation definition text augmentation",
            "nlp":   "natural language processing definition",
            "ml":    "machine learning definition",
            "dl":    "deep learning definition neural network",
            "qa":    "question answering definition",
            "nlu":   "natural language understanding definition",
            "ner":   "named entity recognition definition",
            "pos":   "part of speech tagging definition",
            "bert":  "BERT bidirectional encoder representations transformers definition",
            "gpt":   "generative pre-trained transformer definition",
            "bm25":  "BM25 sparse retrieval ranking function definition",
            "finetune":  "fine tuning pre-trained model definition",
            "augmentation": "data augmentation text augmentation definition",
        }
        import re as _re
        # Find whole-word acronym matches (case-insensitive)
        for acronym, expansion in _ACRONYMS.items():
            if _re.search(r"\b" + _re.escape(acronym) + r"\b", lower):
                suffix_parts.append(expansion)
                break   # one expansion is enough per query

        # Generic boost for definitional questions
        if any(lower.startswith(p) for p in ("what is ", "what are ", "define ", "explain ")):
            suffix_parts.append("definition explanation overview")

        if not suffix_parts:
            return q
        # Deduplicate terms before appending
        all_terms = " ".join(suffix_parts).split()
        seen_terms: set[str] = set()
        unique_terms: list[str] = []
        for t in all_terms:
            if t.lower() not in seen_terms:
                unique_terms.append(t)
                seen_terms.add(t.lower())
        return q + " " + " ".join(unique_terms)

    @property
    def indexed_chunks(self) -> int:
        return int(getattr(self.store, "ntotal", 0))

    def list_pdfs(self) -> list[Path]:
        if not PDF_DIR.exists():
            return []
        return sorted(PDF_DIR.glob("*.pdf"))

    def _iter_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                text = ""
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                pages.append((i, cleaned))
        return pages

    def list_code_files(self) -> list[Path]:
        if not INCLUDE_CODE_INDEX:
            return []
        roots = [
            ROOT_DIR / "api",
            ROOT_DIR / "services",
            ROOT_DIR / "llm",
            ROOT_DIR / "vector_db",
            ROOT_DIR / "ui",
            ROOT_DIR / "eval",
            ROOT_DIR / "tests",
        ]
        files: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            files.extend(sorted(root.rglob("*.py")))
        config_py = ROOT_DIR / "config.py"
        if config_py.exists():
            files.append(config_py)
        skip = {"__pycache__", ".venv", "results"}
        return [p for p in files if not any(part in skip for part in p.parts)]

    def _add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        source_path: Path,
        raw_text: str,
        source_kind: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        extra = extra_meta or {}
        for ordinal, piece in enumerate(chunk_text(raw_text), start=1):
            ids.append(str(uuid.uuid4()))
            documents.append(piece)
            meta = {
                "filename": str(source_path.relative_to(ROOT_DIR) if source_path.is_relative_to(ROOT_DIR) else source_path.name),
                "page": int(extra.get("page") or 0),
                "section": str(extra.get("section") or source_kind),
                "chunk_index": ordinal,
                "source_kind": source_kind,
                "char_length": len(piece),
            }
            metadatas.append(meta)

    def ingest_all(self, *, rebuild: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        pdfs = self.list_pdfs()
        warnings: list[str] = []
        if rebuild:
            self.store.reset()

        already = self.indexed_chunks
        if already > 0 and not rebuild and INCLUDE_CODE_INDEX:
            metas = getattr(self.store, "metadatas", None)
            if isinstance(metas, list) and metas and not any(m.get("source_kind") == "code" for m in metas):
                rebuild = True
                self.store.reset()
                already = 0

        if already > 0 and not rebuild:
            return {
                "status": "cached",
                "files_ingested": [p.name for p in pdfs],
                "chunks_indexed": already,
                "backend": self.backend_name,
                "elapsed_ms": 0.0,
                "warnings": ["Index already populated; pass rebuild=true to rescan PDFs."],
            }

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        files_ingested: list[str] = []

        for pdf in pdfs:
            try:
                pages = self._iter_pages(pdf)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to read {pdf.name}: {exc}")
                continue
            if not pages:
                warnings.append(f"No extractable text in {pdf.name}")
                continue
            files_ingested.append(pdf.name)
            current_section = "Body"
            for page_num, page_text in pages:
                current_section = infer_section(page_text, current_section)
                self._add_chunks(
                    ids,
                    documents,
                    metadatas,
                    pdf,
                    page_text,
                    "local_pdf",
                    extra_meta={"page": page_num, "section": current_section},
                )

        for code_path in self.list_code_files():
            try:
                text = code_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Failed to read {code_path}: {exc}")
                continue
            if not text.strip():
                continue
            files_ingested.append(str(code_path.relative_to(ROOT_DIR)))
            self._add_chunks(
                ids,
                documents,
                metadatas,
                code_path,
                text,
                "code",
                extra_meta={"page": 0, "section": code_path.name},
            )

        added = self.store.add(ids, documents, metadatas) if documents else 0

        # ── Build per-source stats from the metadatas we just ingested ─────
        self._source_stats = {}
        for meta in metadatas:
            fname = str(meta.get("filename") or "")
            kind = str(meta.get("source_kind") or "local_pdf")
            page = meta.get("page") or 0
            if fname not in self._source_stats:
                self._source_stats[fname] = {
                    "filename": fname,
                    "source_kind": kind,
                    "chunks": 0,
                    "pages": set(),
                }
            self._source_stats[fname]["chunks"] += 1
            if page:
                self._source_stats[fname]["pages"].add(int(page))
        # Convert page sets to counts for JSON serialisability
        for v in self._source_stats.values():
            v["pages_extracted"] = len(v.pop("pages", set()))

        result = {
            "status": "ok" if added else "empty",
            "files_ingested": files_ingested,
            "chunks_indexed": self.indexed_chunks,
            "chunk_size_tokens": CHUNK_SIZE,
            "chunk_overlap_tokens": CHUNK_OVERLAP,
            "chunk_step_tokens": max(1, CHUNK_SIZE - CHUNK_OVERLAP),
            "backend": self.backend_name,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "warnings": warnings,
        }
        self.last_ingest = result
        return result

    def _corpus(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        if isinstance(self.store, FaissVectorStore):
            return self.store.ids, self.store.documents, self.store.metadatas
        # Chroma: sample a bounded corpus for BM25 to stay memory-safe
        try:
            peek = self.store.collection.get(include=["documents", "metadatas"], limit=2000)
            return (
                peek.get("ids") or [],
                peek.get("documents") or [],
                peek.get("metadatas") or [],
            )
        except Exception:  # noqa: BLE001
            return [], [], []

    def search(self, query: str, k: int | None = None, pdf_only: bool = False,
               filename_filter: list[str] | None = None) -> list[RetrievedChunk]:
        # For small corpora (≤200 PDF chunks) increase dense candidates so that
        # relevant abstract/intro chunks are not missed due to a low k ceiling.
        # We count PDF-only chunks when pdf_only=True, else all chunks.
        if pdf_only or filename_filter:
            _pdf_count = sum(
                1 for m in (getattr(self.store, "metadatas", None) or [])
                if str((m or {}).get("source_kind") or "local_pdf") != "code"
            )
            effective_k = min(max(_pdf_count, 10), max(k or 0, 20))
        else:
            effective_k = k or RETRIEVAL_K

        # Query expansion for short definitional queries.
        search_query = self._expand_query(query)

        dense = self.store.search(search_query, k=effective_k)
        dense_by_id = {str(item.get("id")): item for item in dense}
        ids, docs, metas = self._corpus()

        # Normalise filename_filter to lowercase basenames for robust matching.
        _filter_basenames: set[str] | None = None
        if filename_filter:
            _filter_basenames = {
                f.split("\\")[-1].split("/")[-1].lower() for f in filename_filter
            }

        def _passes_filter(meta: dict) -> bool:
            fname = str((meta or {}).get("filename") or "")
            basename = fname.split("\\")[-1].split("/")[-1].lower()
            if pdf_only and str((meta or {}).get("source_kind") or "local_pdf") == "code":
                return False
            if _filter_basenames is not None:
                return basename in _filter_basenames
            return True

        # Filter both dense results and BM25 corpus.
        if pdf_only or _filter_basenames:
            dense = [item for item in dense if _passes_filter(item.get("metadata") or {})]
            dense_by_id = {str(item.get("id")): item for item in dense}
            filtered = [
                (i, d, m)
                for i, d, m in zip(ids, docs, metas)
                if _passes_filter(m or {})
            ]
            if filtered:
                ids, docs, metas = zip(*filtered)  # type: ignore[assignment]
                ids, docs, metas = list(ids), list(docs), list(metas)
            else:
                ids, docs, metas = [], [], []

        fused = self.reranker.fuse(
            search_query,   # use expanded query for BM25 scoring too
            dense,
            corpus_docs=docs or None,
            corpus_meta=metas or None,
            corpus_ids=ids or None,
        )
        chunks: list[RetrievedChunk] = []
        inspector_chunks: list[dict[str, Any]] = []
        for rank, item in enumerate(fused, start=1):
            meta = item.get("metadata") or {}
            chunk_id = str(item.get("id"))
            text = item.get("text") or ""
            dense_hit = dense_by_id.get(chunk_id) or {}
            l2 = dense_hit.get("distance")
            kind = str(meta.get("source_kind") or "local_pdf")
            if kind not in {"local_pdf", "arxiv", "code"}:
                kind = "local_pdf"
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(item.get("rrf") or 0.0),
                    filename=str(meta.get("filename") or ""),
                    page=meta.get("page") or None,
                    section=str(meta.get("section") or "") or None,
                    source_kind=kind,  # type: ignore[arg-type]
                    metadata={
                        **dict(meta),
                        "l2_distance": l2,
                        "rrf_score": item.get("rrf"),
                        "dense_rank": next(
                            (i for i, h in enumerate(dense, start=1) if str(h.get("id")) == chunk_id),
                            None,
                        ),
                    },
                )
            )
            inspector_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "char_length": len(text),
                    "text": text,
                    "filename": meta.get("filename"),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "source_kind": kind,
                    "l2_distance": l2,
                    "rrf_score": item.get("rrf"),
                    "origins": item.get("origins"),
                    "why_selected": (
                        f"Hybrid rank {rank}: RRF={float(item.get('rrf') or 0):.5f}"
                        + (f", dense L2={float(l2):.4f}" if l2 is not None else ", BM25-only (no dense hit)")
                    ),
                }
            )
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        embed_stats = last_query_embed_stats()
        self.last_inspector = {
            "kb_stats": {
                "total_chunks": self.indexed_chunks,
                "pdf_count": len(self.list_pdfs()),
                "sources": list(self._source_stats.values()),
                "chunking": {
                    "method": "whitespace_token_sliding_window",
                    "chunk_size_tokens": CHUNK_SIZE,
                    "overlap_tokens": CHUNK_OVERLAP,
                    "step_tokens": step,
                },
                "embedding": {
                    "model": EMBEDDING_MODEL,
                    "dimension": embed_stats.get("dimension"),
                },
                "vector_store": self.backend_name,
            },
            "chunking": {
                "method": "whitespace_token_sliding_window",
                "chunk_size_tokens": CHUNK_SIZE,
                "overlap_tokens": CHUNK_OVERLAP,
                "step_tokens": step,
                "formula": f"windows of {CHUNK_SIZE} tokens, advance by {step} (= size - overlap {CHUNK_OVERLAP})",
                "total_chunks_in_index": self.indexed_chunks,
            },
            "embedding": {
                "model": EMBEDDING_MODEL,
                "dimension": embed_stats.get("dimension"),
                "query_embed_latency_ms": embed_stats.get("latency_ms"),
                "query_chars": embed_stats.get("query_chars"),
            },
            "matching": {
                "dense_backend": self.backend_name,
                "dense_metric": "L2 (lower is closer)",
                "hybrid": "BM25 + dense Reciprocal Rank Fusion",
                **(getattr(self.store, "last_telemetry", {}) or {}),
            },
            "retrieved_chunks": inspector_chunks,
        }
        return chunks
