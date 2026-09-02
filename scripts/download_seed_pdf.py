"""Download a small seed academic PDF into data/pdfs/rag_survey.pdf."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "pdfs" / "rag_survey.pdf"
# Lewis et al. 2020 RAG paper — compact, well-known seed document
URL = "https://arxiv.org/pdf/2005.11401.pdf"
UA = "AcademicResearchAssistant/1.0 (local seed download)"


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 10_000:
        print(f"Already present: {TARGET}")
        return 0
    request = Request(URL, headers={"User-Agent": UA})
    with urlopen(request, timeout=30) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        print("Download did not return a PDF", file=sys.stderr)
        return 1
    TARGET.write_bytes(data)
    print(f"Wrote {TARGET} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
