"""Web search tool: search the web and optionally scrape top results for context."""

import re
import warnings
from src.tool import Tool

try:
    from ddgs import DDGS
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            DDGS = None

try:
    import requests
except ImportError:
    requests = None

_DEFAULT_MAX_RESULTS = 5
_MAX_SCRAPE_CHARS = 4000
_REQUEST_TIMEOUT = 10


def _alternative_queries(query: str) -> list[str]:
    """Build fallback queries when the original returns no results (shorter, key terms)."""
    words = [w for w in query.split() if len(w) > 1]
    if not words:
        return []
    alternatives = []
    if len(words) > 4:
        alternatives.append(" ".join(words[:4]))
    if len(words) > 2:
        alternatives.append(" ".join(words[:2]))
    if len(words) > 1:
        alternatives.append(words[0])
    return alternatives[:3]


def _extract_text_from_url(url: str, max_chars: int = _MAX_SCRAPE_CHARS) -> str:
    """Fetch URL and extract plain text. Returns truncated text or error message."""
    if not requests:
        return "(Install requests to scrape page content.)"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; CouncilBot/1.0; +https://github.com/ai-council)",
        }
        r = requests.get(url, timeout=_REQUEST_TIMEOUT, headers=headers)
        r.raise_for_status()
        text = r.text
        if "<" in text and ">" in text:
            start = text.find("<body")
            if start != -1:
                end = text.find("</body>", start)
                end = end + 7 if end != -1 else len(text)
                text = text[start:end]
            text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        text = (text or "").strip() or "(No text extracted.)"
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text.strip() or "(No text extracted.)"
    except Exception as e:
        return f"(Could not fetch: {e})"


class SearchTool(Tool):
    """Search the web for market research, facts, or information. Use for data the model lacks."""

    name = "search"
    description = (
        "Search the web for current information, market research, or facts. "
        "Returns titles, snippets, and URLs; optionally scrapes top result pages for more detail."
    )
    parameters = (
        "query (str): search query; "
        "max_results (int, optional): number of results to return, default 5; "
        "scrape_top (int, optional): number of top results to fetch and extract text from, 0-2, default 0"
    )

    def __init__(self, max_results: int = _DEFAULT_MAX_RESULTS, scrape_top: int = 0):
        if DDGS is None:
            raise ImportError("Install ddgs for web search: pip install ddgs")
        self._default_max_results = max(1, min(max_results, 10))
        self._scrape_top = max(0, min(scrape_top, 2))

    def _do_search(self, query: str, n: int) -> list[dict]:
        """Run one search; suppresses duckduckgo_search rename warning if present."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return list(DDGS().text(query, max_results=n))

    def run(
        self,
        query: str,
        max_results: int | None = None,
        scrape_top: int | None = None,
    ) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: query cannot be empty."
        n = max_results if max_results is not None else self._default_max_results
        n = max(1, min(n, 10))
        do_scrape = scrape_top if scrape_top is not None else self._scrape_top
        do_scrape = max(0, min(do_scrape, 2))

        try:
            results = self._do_search(query, n)
            used_query = query
            if not results:
                for alt in _alternative_queries(query):
                    if alt and alt != query:
                        results = self._do_search(alt, n)
                        if results:
                            used_query = alt
                            break
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            tried = [query] + [q for q in _alternative_queries(query) if q and q != query]
            return (
                f"No results found for: \"{query}\". "
                f"Tried fallback queries: {tried}. "
                "Suggest rephrasing with simpler or fewer terms, or different keywords."
            )

        lines = [f"Search results for: {used_query}", ""]
        if used_query != query:
            lines.append(f"(Original query \"{query}\" had no results; showing results for shorter query.)")
            lines.append("")
        for i, r in enumerate(results):
            title = r.get("title", "")
            href = r.get("href", r.get("link", ""))
            body = r.get("body", r.get("snippet", ""))
            lines.append(f"{i + 1}. {title}")
            lines.append(f"   URL: {href}")
            lines.append(f"   {body}")
            lines.append("")

        if do_scrape > 0 and results:
            lines.append("--- Scraped content from top result(s) ---")
            for i, r in enumerate(results[:do_scrape]):
                url = r.get("href", r.get("link", ""))
                if not url:
                    continue
                lines.append(f"\nFrom: {url}")
                lines.append(_extract_text_from_url(url))
                lines.append("")

        return "\n".join(lines).strip()
