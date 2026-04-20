"""LLM lookup of top competitors to expand keyword news search (NewsAPI, GNews, Exa, Jina)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.config import settings
from app.services.llm_article_classifier import _call_openai_json

logger = logging.getLogger(__name__)

_CACHE: dict[str, list[str]] = {}
_CACHE_MAX = 300


def news_competitor_llm_enabled() -> bool:
    """Default on when OPENAI_API_KEY is set; override with NEWS_COMPETITOR_LLM_ENABLED=false."""
    raw = os.getenv("NEWS_COMPETITOR_LLM_ENABLED")
    if raw is None:
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _sanitize_competitor_name(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    s = re.sub(r"[^\w\s&\-\./]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:80]


def _system_prompt() -> str:
    return (
        "You identify the main business competitors of a public company for equity research. "
        'Respond ONLY with valid JSON: {"competitors": ["...", ...]} with at most 5 items. '
        "Use company names or tickers as they typically appear in financial headlines. "
        "No commentary. If the issuer is unclear, return an empty list or fewer than 5 names."
    )


def _user_prompt(ticker: str, info: dict | None) -> str:
    lines = [f"Ticker: {ticker.upper()}"]
    if info:
        for key in ("longName", "shortName", "sector", "industry"):
            v = info.get(key)
            if isinstance(v, str) and v.strip():
                lines.append(f"{key}: {v.strip()}")
    return "\n".join(lines) + '\n\nReturn JSON with key "competitors" listing up to 5 top competitors.'


def top_competitors_for_news_search(ticker: str, info: dict | None) -> list[str]:
    """Return up to 5 competitor names/tickers for search query expansion; [] on skip or failure."""
    sym = ticker.strip().upper()
    if not sym or not news_competitor_llm_enabled():
        return []
    if sym in _CACHE:
        cached = _CACHE[sym]
        logger.info(
            "Competitors for news search ticker=%s names=%s (cached)",
            sym,
            cached if cached else "(none)",
        )
        return cached

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return []

    try:
        parsed: dict[str, Any] = _call_openai_json(
            api_key=api_key,
            base_url=settings.openai_base_url,
            model=settings.news_llm_model,
            system_text=_system_prompt(),
            user_text=_user_prompt(ticker, info),
            timeout=settings.news_competitor_llm_timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Competitor LLM lookup failed for %s: %s", sym, exc)
        return []

    raw = parsed.get("competitors")
    if not isinstance(raw, list):
        out: list[str] = []
    else:
        seen: set[str] = set()
        out = []
        for x in raw:
            if not isinstance(x, str):
                continue
            t = _sanitize_competitor_name(x)
            if not t:
                continue
            low = t.lower()
            if low in seen or low == sym.lower():
                continue
            seen.add(low)
            out.append(t)
            if len(out) >= 5:
                break

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[sym] = out
    logger.info(
        "Competitors for news search ticker=%s names=%s",
        sym,
        out if out else "(none)",
    )
    return out
