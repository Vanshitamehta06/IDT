"""PDF library listing, upload, and deletion helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from config import FAISS_META_PATH, PDF_DIR, ensure_data_dirs


def _page_count(path: Path) -> int | None:
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return None


def list_library(*, indexed_filenames: set[str] | None = None) -> list[dict]:
    ensure_data_dirs()
    index_mtime = FAISS_META_PATH.stat().st_mtime if FAISS_META_PATH.exists() else 0.0
    indexed = indexed_filenames or set()
    items: list[dict] = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        mtime = pdf.stat().st_mtime
        in_index = pdf.name in indexed if indexed else bool(index_mtime and mtime <= index_mtime + 2)
        if indexed:
            status = "indexed" if in_index else "needs_reindex"
        else:
            status = "indexed" if index_mtime and mtime <= index_mtime + 2 else "needs_reindex"
        items.append(
            {
                "filename": pdf.name,
                "size_bytes": pdf.stat().st_size,
                "pages": _page_count(pdf),
                "last_modified": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                "status": status,
            }
        )
    return items


def save_pdf(filename: str, data: bytes) -> Path:
    ensure_data_dirs()
    safe = Path(filename).name
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    path = PDF_DIR / safe
    path.write_bytes(data)
    return path


def delete_pdf(filename: str) -> bool:
    path = PDF_DIR / Path(filename).name
    if not path.exists() or path.suffix.lower() != ".pdf":
        return False
    path.unlink()
    return True
