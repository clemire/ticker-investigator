"""Batched LLM classification for news articles (non-naive: chunked prompts, cache, fallback)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import httpx

from app.models import NewsArticle

logger = logging.getLogger(__name__)

Category = Literal["company", "competitor", "industry", "macro", "unknown"]

_CACHE: dict[str, Category] = {}
_CACHE_MAX = 2000


def news_llm_classification_enabled() -> bool:
    """Default on when OPENAI_API_KEY is set; override with NEWS_LLM_CLASSIFIER_ENABLED=false."""
    raw = os.getenv("NEWS_LLM_CLASSIFIER_ENABLED")
    if raw is None:
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _cache_key(ticker: str, article: NewsArticle) -> str:
    return hashlib.sha256(
        f"{ticker.upper()}|{article.url}|{article.title}".encode(),
        usedforsecurity=False,
    ).hexdigest()


def _trim(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


def _system_prompt() -> str:
    return (
        "You classify financial news headlines/snippets for equity research. "
        "Respond ONLY with valid JSON matching the schema in the user message.\n"
        "Definitions:\n"
        "- company: primarily about this issuer—earnings, products, management, legal/regulatory action against the firm, "
        "firm-specific stock moves, analyst price targets on this name, OR pages clearly centered on this ticker "
        "(e.g. title contains the company name and/or ticker like 'Apple (AAPL)', stock analysis, financial data for this symbol). "
        "Those are company, not unknown.\n"
        "- competitor: the headline clearly centers on a named peer/rival of the issuer (another company or its ticker), not primarily the issuer itself—e.g. earnings, stock move, or major news about that peer when the research ticker is different.\n"
        "- industry: sector/peers/competition/supply chain/market structure affecting a group of companies; not mainly one company’s idiosyncratic story.\n"
        "- macro: rates, inflation, Fed/policy, regulation at economy-wide level, elections, geopolitics, tariffs/sanctions when not firm-specific.\n"
        "- unknown: only when the subject company/ticker cannot be determined or the text is too vague—do not use unknown when the title names the issuer or ticker."
    )


def reconcile_llm_unknown_with_keyword_company(
    articles: list[NewsArticle],
    ticker: str,
    info: dict | None,
    keyword_classify: Callable[[str, str, dict | None], str],
) -> list[NewsArticle]:
    """If the model returns unknown but deterministic rules say company, prefer company (and refresh cache)."""
    out: list[NewsArticle] = []
    for art in articles:
        if art.category != "unknown":
            out.append(art)
            continue
        body = f"{art.title}\n{art.description or ''}"
        kw = keyword_classify(body, ticker, info)
        if kw == "company":
            _cache_put(_cache_key(ticker, art), "company")
            out.append(art.model_copy(update={"category": "company"}))
        elif kw == "competitor":
            _cache_put(_cache_key(ticker, art), "competitor")
            out.append(art.model_copy(update={"category": "competitor"}))
        else:
            out.append(art)
    return out


def _user_payload_chunk(
    ticker: str,
    company_label: str,
    chunk: list[tuple[int, NewsArticle]],
) -> str:
    lines = [
        f"Ticker: {ticker.upper()}",
        f"Issuer / company context: {company_label or '(not provided)'}",
        "",
        "Classify each article below. Return JSON exactly in this shape:",
        '{"results":[{"index":0,"category":"company|competitor|industry|macro|unknown"}, ...]}',
        "Indices are 0-based within this batch only.",
        "",
        "Articles:",
    ]
    for local_i, art in chunk:
        lines.append(f"[{local_i}] Title: {_trim(art.title, 240)}")
        lines.append(f"    Snippet: {_trim(art.description, 400)}")
        lines.append("")
    return "\n".join(lines)


def _call_openai_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_text: str,
    user_text: str,
    timeout: float,
) -> dict:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code == 400:
            payload.pop("response_format", None)
            r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    content = data["choices"][0]["message"]["content"]
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Some OpenAI-compatible servers return JSON without json_object mode.
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _classify_chunk(
    *,
    ticker: str,
    company_label: str,
    indexed_articles: list[tuple[int, NewsArticle]],
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> dict[int, Category]:
    """Local index within chunk -> category."""
    user_text = _user_payload_chunk(ticker, company_label, indexed_articles)
    try:
        parsed = _call_openai_json(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_text=_system_prompt(),
            user_text=user_text,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("LLM classification chunk failed for %s: %s", ticker, exc)
        return {}

    out: dict[int, Category] = {}
    results = parsed.get("results")
    if not isinstance(results, list):
        return {}
    allowed: set[str] = {"company", "competitor", "industry", "macro", "unknown"}
    for row in results:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row["index"])
        except (KeyError, TypeError, ValueError):
            continue
        cat = row.get("category", "unknown")
        if isinstance(cat, str) and cat in allowed:
            out[idx] = cat  # type: ignore[assignment]
    return out


def _cache_put(key: str, value: Category) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = value


def apply_llm_categories(
    articles: list[NewsArticle],
    ticker: str,
    info: dict | None,
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    batch_size: int,
    timeout: float,
    max_parallel_chunks: int = 3,
) -> list[NewsArticle]:
    """Apply batched LLM labels in parallel across chunks; cache by URL+title; keep keyword category on failure."""
    if not articles:
        return articles
    if not news_llm_classification_enabled():
        return articles

    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        logger.info("Skipping LLM article classification: no OPENAI_API_KEY")
        return articles

    company_label = ""
    if info:
        raw = info.get("longName") or info.get("shortName") or info.get("displayName")
        if isinstance(raw, str):
            company_label = raw.strip()

    # Positions that still need an LLM call (cache miss)
    need_llm: list[tuple[int, NewsArticle]] = []
    resolved: dict[int, Category] = {}

    for i, art in enumerate(articles):
        ck = _cache_key(ticker, art)
        if ck in _CACHE:
            resolved[i] = _CACHE[ck]
        else:
            need_llm.append((i, art))

    if need_llm:
        chunks: list[list[tuple[int, NewsArticle]]] = []
        for offset in range(0, len(need_llm), batch_size):
            batch = need_llm[offset : offset + batch_size]
            chunks.append([(j, batch[j][1]) for j in range(len(batch))])

        def run_chunk(chunk: list[tuple[int, NewsArticle]]) -> dict[int, Category]:
            return _classify_chunk(
                ticker=ticker,
                company_label=company_label,
                indexed_articles=chunk,
                api_key=key,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )

        chunk_results: list[tuple[int, dict[int, Category]]] = []
        if len(chunks) == 1:
            chunk_results.append((0, run_chunk(chunks[0])))
        else:
            with ThreadPoolExecutor(max_workers=min(max_parallel_chunks, len(chunks))) as pool:
                future_to_idx = {pool.submit(run_chunk, ch): idx for idx, ch in enumerate(chunks)}
                for fut in as_completed(future_to_idx):
                    chunk_idx = future_to_idx[fut]
                    try:
                        m = fut.result()
                    except Exception as exc:
                        logger.warning("LLM chunk executor failed: %s", exc)
                        continue
                    chunk_results.append((chunk_idx, m))

        for chunk_idx, mapping in sorted(chunk_results, key=lambda x: x[0]):
            offset = chunk_idx * batch_size
            batch = need_llm[offset : offset + batch_size]
            for local_i, cat in mapping.items():
                if local_i < 0 or local_i >= len(batch):
                    continue
                global_i = batch[local_i][0]
                resolved[global_i] = cat
                _cache_put(_cache_key(ticker, articles[global_i]), cat)

    out: list[NewsArticle] = []
    for i, art in enumerate(articles):
        if i in resolved:
            out.append(art.model_copy(update={"category": resolved[i]}))
        else:
            out.append(art)
    return out
