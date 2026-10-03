"""Web tools available to the swarm: search and scrape, no API keys needed.

The scout searches DuckDuckGo's HTML endpoint; the scraper fetches pages and
extracts readable text. Both are polite: real User-Agent, short timeouts,
strict size caps.
"""

from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "crewSwarm/0.1 (+https://localhost; research bot)"
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

MAX_PAGE_BYTES = 1_500_000
TEXT_CHARS = 6_000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web, no API key. Tries the `ddgs` client first and falls
    back to DuckDuckGo's HTML endpoint. Returns [] on any failure — the
    swarm treats a failed search as a weak pheromone, not an error."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            rows = ddgs.text(query, max_results=max_results)
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href", ""),
                         snippet=r.get("body", ""))
            for r in rows
            if r.get("href", "").startswith("http")
        ][:max_results]
    except Exception:
        pass
    return _search_ddg_html(query, max_results)


def _search_ddg_html(query: str, max_results: int = 5) -> list[SearchResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = SESSION.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    html = resp.text
    results: list[SearchResult] = []
    # result links: <a class="result__a" href="...">title</a>
    link_re = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.S,
    )
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)

    links = list(link_re.finditer(html))
    snippets = list(snippet_re.finditer(html))
    for i, m in enumerate(links[:max_results]):
        href = m.group("href")
        # duckduckgo wraps urls as //duckduckgo.com/l/?uddg=<encoded>
        udg = re.search(r"uddg=([^&]+)", href)
        if udg:
            from urllib.parse import unquote

            href = unquote(udg.group(1))
        title = re.sub(r"<[^>]+>", "", m.group("title")).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r"<[^>]+>", "", snippets[i].group(1)).strip()
        if href.startswith("http"):
            results.append(SearchResult(title=title, url=href, snippet=snippet))
    return results


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def scrape_url(url: str, max_chars: int = TEXT_CHARS) -> str:
    """Fetch a page and return cleaned main-ish text."""
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        resp = SESSION.get(url, timeout=12, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""
        chunks, size = [], 0
        for chunk in resp.iter_content(65536, decode_unicode=False):
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_PAGE_BYTES:
                break
        raw = b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore")
    except requests.RequestException:
        return ""

    text = _SCRIPT_RE.sub(" ", raw)
    # drop nav/footer-ish boilerplate tags entirely
    text = re.sub(r"<(nav|footer|header|aside)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


def key_terms(question: str, n: int = 6) -> list[str]:
    """Cheap keyword extraction used to aim the scout's searches and to
    score pheromone relevance."""
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
        "will", "would", "could", "should", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "of", "to",
        "in", "on", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "from", "up",
        "down", "out", "off", "over", "under", "again", "further", "once",
        "here", "there", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "can", "just", "what", "which", "who",
        "whom", "this", "that", "these", "those", "am", "it", "its", "i", "you",
        "he", "she", "we", "they", "them", "his", "her", "their", "our", "your",
        "my", "me", "him", "us", "how", "why", "where", "as", "at", "by",
        "predict", "happens", "happen", "2026", "2025",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'\-]{2,}", question.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:n]]


def polite_sleep(seconds: float = 0.4) -> None:
    time.sleep(seconds)
