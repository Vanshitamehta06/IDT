"""arXiv Atom API client with timeouts, encoding, and graceful failures."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote_plus
from urllib.request import Request, urlopen

from api.schemas import ArxivPaper
from config import ARXIV_BASE_URL, ARXIV_MAX_RESULTS, ARXIV_TIMEOUT_SECONDS, USER_AGENT
from llm.prompt_builder import bound_text

logger = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


class DiscoveryService:
    def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        start: int = 0,
    ) -> tuple[list[ArxivPaper], dict]:
        q = (query or "").strip()
        telemetry: dict = {"provider": "arxiv", "ok": False, "query": q}
        if not q:
            telemetry["error"] = "empty_query"
            return [], telemetry

        params = {
            "search_query": f"all:{q}",
            "start": start,
            "max_results": max_results or ARXIV_MAX_RESULTS,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_BASE_URL}?{urlencode(params, quote_via=quote_plus)}"
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=ARXIV_TIMEOUT_SECONDS) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("arXiv search failed: %s", exc)
            telemetry["error"] = str(exc)
            return [], telemetry

        papers = self._parse_atom(payload)
        telemetry.update({"ok": True, "returned": len(papers), "url": url})
        return papers, telemetry

    def _parse_atom(self, payload: bytes) -> list[ArxivPaper]:
        root = ET.fromstring(payload)
        papers: list[ArxivPaper] = []
        for entry in root.findall(f"{ATOM}entry"):
            arxiv_id = _text(entry.find(f"{ATOM}id")).rsplit("/", 1)[-1]
            title = _text(entry.find(f"{ATOM}title"))
            summary = bound_text(_text(entry.find(f"{ATOM}summary")), 1200)
            published = _text(entry.find(f"{ATOM}published"))
            updated = _text(entry.find(f"{ATOM}updated"))
            authors = [
                _text(author.find(f"{ATOM}name"))
                for author in entry.findall(f"{ATOM}author")
            ]
            categories = [
                (cat.get("term") or "")
                for cat in entry.findall(f"{ATOM}category")
                if cat.get("term")
            ]
            links = entry.findall(f"{ATOM}link")
            abs_url = _text(entry.find(f"{ATOM}id"))
            for link in links:
                if link.get("rel") == "alternate" and link.get("href"):
                    abs_url = link.get("href") or abs_url
            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=[a for a in authors if a],
                    abstract=summary,
                    published=published,
                    updated=updated,
                    url=abs_url,
                    categories=categories,
                )
            )
        return papers
