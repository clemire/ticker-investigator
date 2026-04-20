from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import logging
import os
import re
from xml.etree import ElementTree

import httpx
import yfinance as yf
from dateutil import parser as date_parser

from app.config import settings
from app.models import NewsArticle
from app.services.llm_article_classifier import (
    apply_llm_categories,
    reconcile_llm_unknown_with_keyword_company,
)
from app.services.ticker_competitors import top_competitors_for_news_search

JINA_NEWS_ENDPOINT = "https://s.jina.ai/http://news.google.com/rss/search"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"
GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
EXA_ENDPOINT = "https://api.exa.ai/search"
logger = logging.getLogger(__name__)

# Tokens stripped from company names when building keyword matches (e.g. "Apple" from "Apple Inc.")
_COMPANY_STOPWORDS = frozenset(
    {
        "inc",
        "corp",
        "corporation",
        "ltd",
        "limited",
        "plc",
        "nv",
        "sa",
        "the",
        "company",
        "holdings",
        "group",
        "co",
        "llc",
        "class",
        "ordinary",
        "adr",
    }
)


def _build_company_terms(ticker: str, info: dict | None) -> list[str]:
    """Terms used to detect company-specific news. Headlines usually say 'Apple', not 'AAPL'."""
    terms: set[str] = set()
    sym = ticker.strip()
    if sym:
        terms.add(sym.lower())
    if not info:
        return sorted(terms)
    for key in ("symbol", "shortName", "longName", "displayName"):
        val = info.get(key)
        if not isinstance(val, str):
            continue
        s = val.strip()
        if not s:
            continue
        terms.add(s.lower())
        for word in re.split(r"[^\w]+", s):
            w = word.lower().strip()
            if len(w) >= 3 and w not in _COMPANY_STOPWORDS:
                terms.add(w)
    return sorted(terms)


def _to_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return date_parser.parse(value)
        except (ValueError, TypeError):
            return None
    return None


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# Shared with `_classify_article` and `_compute_relevance_score` so industry/macro stories
# without the ticker in the headline still get a relevance boost (otherwise they were dropped).
_INDUSTRY_KEYWORDS = (
    "competitor",
    "competition",
    "sector",
    "industry",
    "market share",
    "peer",
    "supply chain",
)
_MACRO_KEYWORDS = (
    "federal reserve",
    "the fed",
    "interest rate",
    "rates decision",
    "inflation",
    "geopolitical",
    "geopolitics",
    "regulation",
    "regulatory",
    "tariff",
    "sanctions",
    "election",
    "congress",
    "white house",
    "trade war",
    "ftc",
    "antitrust",
    "subpoena",
)


def _in_date_range(published_at: datetime | None, start_date: date, end_date: date) -> bool:
    if published_at is None:
        return True
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    return start_dt <= _ensure_utc(published_at) < end_dt


def _classify_article(text: str, ticker: str, info: dict | None = None) -> str:
    lower = text.lower()
    company_terms = _build_company_terms(ticker, info)
    if any(term and term in lower for term in company_terms):
        return "company"
    if any(term in lower for term in _INDUSTRY_KEYWORDS):
        return "industry"
    if any(term in lower for term in _MACRO_KEYWORDS):
        return "macro"
    return "unknown"


def _compute_relevance_score(text: str, ticker: str, info: dict | None) -> float:
    """Post-fetch relevance in [0, 1]: ticker/company overlap, plus thematic industry/macro signals.

    Without the latter, purely macro or sector stories (no symbol in headline) scored ~0 and were
    dropped before classification—same keyword lists as `_classify_article` for consistency.
    """
    t = text.lower()
    sym_l = ticker.strip().lower()
    best = 0.0
    for term in _build_company_terms(ticker, info):
        if not term or len(term) < 2:
            continue
        if len(term) < 3:
            if re.search(rf"\b{re.escape(term)}\b", t):
                best = max(best, 0.32)
            continue
        if term not in t:
            continue
        w = min(len(term), 14)
        best = max(best, 0.36 + 0.035 * (w - 2))
    if sym_l and len(sym_l) >= 2:
        if re.search(rf"\b{re.escape(sym_l)}\b", t):
            best = max(best, 0.52)
        elif len(sym_l) >= 4 and sym_l in t:
            best = max(best, 0.48)
    if sym_l and (
        f"${sym_l}" in t
        or f"({sym_l})" in t
        or f":{sym_l}" in t
        or f"/{sym_l}" in t
        or f"nasdaq:{sym_l}" in t
        or f"nyse:{sym_l}" in t
    ):
        best = max(best, 0.58)
    # Keep industry / macro / “competitor context” pieces that never mention the ticker by name.
    if any(kw in t for kw in _INDUSTRY_KEYWORDS):
        best = max(best, 0.44)
    if any(kw in t for kw in _MACRO_KEYWORDS):
        best = max(best, 0.44)
    return min(best, 1.0)


def _finalize_news_with_relevance(
    articles: list[NewsArticle],
    ticker: str,
    info: dict | None,
    *,
    min_score: float,
    limit: int,
) -> list[NewsArticle]:
    deduped = _dedupe_articles(articles)
    enriched: list[NewsArticle] = []
    for item in deduped:
        body = f"{item.title}\n{item.description or ''}"
        category = item.category
        if category == "unknown":
            category = _classify_article(body, ticker, info)
        score = _compute_relevance_score(body, ticker, info)
        enriched.append(
            item.model_copy(
                update={
                    "category": category,
                    "relevance_score": round(score, 4),
                }
            )
        )
    relevant = [a for a in enriched if a.relevance_score >= min_score]
    relevant.sort(key=lambda a: (-a.relevance_score, a.title))
    dropped = len(enriched) - len(relevant)

    for a in enriched:
        if a.relevance_score < min_score:
            logger.info(
                "news_drop_relevance ticker=%s score=%.4f threshold=%.4f title=%r url=%s",
                ticker,
                a.relevance_score,
                min_score,
                (a.title or "")[:280],
                a.url,
            )
    if dropped:
        logger.info(
            "Relevance filter: dropped %s/%s articles below %.2f for %s",
            dropped,
            len(enriched),
            min_score,
            ticker,
        )
    if not relevant and enriched:
        logger.warning(
            "All %s articles below relevance threshold %.2f for %s",
            len(enriched),
            min_score,
            ticker,
        )
    sliced = relevant[:limit]
    if len(relevant) > limit:
        for a in relevant[limit:]:
            logger.info(
                "news_drop_limit ticker=%s news_limit=%s score=%.4f title=%r url=%s",
                ticker,
                limit,
                a.relevance_score,
                (a.title or "")[:280],
                a.url,
            )

    sliced = apply_llm_categories(
        sliced,
        ticker,
        info,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=settings.openai_base_url,
        model=settings.news_llm_model,
        batch_size=settings.news_llm_batch_size,
        timeout=settings.news_llm_timeout_seconds,
        max_parallel_chunks=settings.news_llm_max_parallel_chunks,
    )
    return reconcile_llm_unknown_with_keyword_company(sliced, ticker, info, _classify_article)


def _dedupe_articles(articles: list[NewsArticle]) -> list[NewsArticle]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    deduped: list[NewsArticle] = []
    for article in articles:
        t = article.title.strip().lower()
        if article.url in seen_urls or t in seen_titles:
            continue
        seen_urls.add(article.url)
        seen_titles.add(t)
        deduped.append(article)
    return deduped


def _yfinance_stream_item_parse(
    item: dict,
) -> tuple[str, str, str | None, datetime | None, str] | None:
    """Map a Yahoo tickerStream row to (title, url, summary, published_at, source).

    Yahoo has returned flat fields; newer responses nest story data under ``content``
    (title, pubDate ISO string, canonicalUrl.url, provider.displayName).
    """
    c = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = (item.get("title") or c.get("title") or "").strip()
    summary = item.get("summary")
    if summary is None:
        summary = c.get("summary") or c.get("description")
    if isinstance(summary, str):
        summary = summary.strip() or None

    url = item.get("link") or item.get("url")
    if not url:
        for key in ("canonicalUrl", "clickThroughUrl"):
            block = c.get(key)
            if isinstance(block, dict):
                u = block.get("url")
                if u:
                    url = u
                    break
    if not url and isinstance(c.get("canonicalUrl"), str):
        url = c["canonicalUrl"]

    published_at: datetime | None = None
    pub_ts = item.get("providerPublishTime")
    if pub_ts is not None:
        try:
            published_at = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
        except (OSError, ValueError, OverflowError, TypeError):
            published_at = None
    if published_at is None:
        for key in ("pubDate", "displayTime"):
            raw = c.get(key) or item.get(key)
            if isinstance(raw, str) and raw.strip():
                published_at = _ensure_utc(_to_datetime(raw.strip()))
                if published_at:
                    break

    src = str(item.get("publisher") or "").strip()
    if not src and isinstance(c.get("provider"), dict):
        src = str(c["provider"].get("displayName") or "").strip()
    source = src or "yfinance"

    if not title or not url:
        return None
    return (title, str(url), summary, published_at, source)


def fetch_company_news_yfinance(ticker: str, start_date: date, end_date: date, limit: int) -> list[NewsArticle]:
    logger.info("fetch_company_news_yfinance ticker=%s start_date=%s end_date=%s limit=%s", ticker, start_date, end_date, limit)
    t = yf.Ticker(ticker)
    try:
        info = t.info if isinstance(t.info, dict) else {}
    except Exception:
        info = {}
    raw_news = t.news or []
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    articles: list[NewsArticle] = []
    logger.info("yfinance news ticker=%s raw_items=%s (window %s..%s, limit=%s)", ticker, len(raw_news), start_date, end_date, limit)
    for item in raw_news:
        if not isinstance(item, dict):
            continue
        parsed = _yfinance_stream_item_parse(item)
        if not parsed:
            continue
        title, url, summary, published_at, source = parsed
        if published_at and not (start_dt <= published_at < end_dt):
            continue

        category = _classify_article(f"{title}\n{summary or ''}", ticker, info)
        articles.append(
            NewsArticle(
                title=title,
                source=source,
                published_at=published_at,
                url=url,
                description=summary,
                category=category,
            )
        )

    articles.sort(key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    out = _dedupe_articles(articles)[:limit]
    logger.info(
        "yfinance news ticker=%s raw_items=%s returned=%s (window %s..%s, limit=%s)",
        ticker,
        len(raw_news),
        len(out),
        start_date,
        end_date,
        limit,
    )
    return out


def _parse_rss_articles(
    rss_xml: str,
    *,
    source_name: str,
    ticker: str,
    info: dict | None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[NewsArticle]:
    try:
        root = ElementTree.fromstring(rss_xml)
    except ElementTree.ParseError:
        return []

    start_dt: datetime | None = None
    end_dt: datetime | None = None
    if start_date is not None and end_date is not None:
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    items = root.findall(".//item")
    articles: list[NewsArticle] = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip() or None
        published_at: datetime | None = None

        pub_date = (item.findtext("pubDate") or "").strip()
        if pub_date:
            try:
                published_at = _ensure_utc(parsedate_to_datetime(pub_date))
            except (TypeError, ValueError):
                published_at = None

        if start_dt and end_dt and published_at and not (start_dt <= published_at < end_dt):
            continue
        if not title or not link:
            continue

        category = _classify_article(f"{title}\n{description or ''}", ticker, info)
        articles.append(
            NewsArticle(
                title=title,
                source=source_name,
                published_at=published_at,
                url=link,
                description=description,
                category=category,
            )
        )
    return articles


def fetch_general_news_jina(query: str, limit: int, timeout_seconds: float) -> list[NewsArticle]:
    # Jina AI hosted fetch for Google News RSS is a simple no-key fallback.
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(JINA_NEWS_ENDPOINT, params=params)
        response.raise_for_status()
        text = response.text

    articles: list[NewsArticle] = []
    for line in text.splitlines():
        # The Jina wrapper returns markdown-like rendering; parse links heuristically.
        if "](" not in line:
            continue
        if line.strip().startswith("- ["):
            try:
                title = line.split("- [", 1)[1].split("](", 1)[0].strip()
                url = line.split("](", 1)[1].rsplit(")", 1)[0].strip()
            except (IndexError, ValueError):
                continue
            if not title or not url:
                continue
            articles.append(
                NewsArticle(
                    title=title,
                    source="Google News",
                    published_at=None,
                    url=url,
                    description=None,
                    category="unknown",
                )
            )
        if len(articles) >= limit:
            break
    return _dedupe_articles(articles)


def fetch_newsapi_news(
    query: str,
    *,
    ticker: str,
    info: dict | None,
    start_date: date,
    end_date: date,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    api_key = os.getenv("NEWSAPI_API_KEY")
    if not api_key:
        return []
    params = {
        "q": query,
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(100, max(1, limit)),
    }
    headers = {"X-Api-Key": api_key}
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(NEWSAPI_ENDPOINT, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()

    articles: list[NewsArticle] = []
    for item in payload.get("articles", []):
        title = item.get("title")
        url = item.get("url")
        if not title or not url:
            continue
        published_at = _to_datetime(item.get("publishedAt"))
        if not _in_date_range(published_at, start_date, end_date):
            continue
        description = item.get("description")
        category = _classify_article(f"{title}\n{description or ''}", ticker, info)
        articles.append(
            NewsArticle(
                title=title,
                source=str((item.get("source") or {}).get("name") or "NewsAPI"),
                published_at=published_at,
                url=url,
                description=description,
                category=category,
            )
        )
    return _dedupe_articles(articles)[:limit]


def fetch_gnews_news(
    query: str,
    *,
    ticker: str,
    info: dict | None,
    start_date: date,
    end_date: date,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []
    params = {
        "q": query,
        "from": f"{start_date.isoformat()}T00:00:00Z",
        "to": f"{end_date.isoformat()}T23:59:59Z",
        "lang": "en",
        "max": min(100, max(1, limit)),
        "token": api_key,
    }
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(GNEWS_ENDPOINT, params=params)
        response.raise_for_status()
        payload = response.json()

    articles: list[NewsArticle] = []
    for item in payload.get("articles", []):
        title = item.get("title")
        url = item.get("url")
        if not title or not url:
            continue
        published_at = _to_datetime(item.get("publishedAt"))
        if not _in_date_range(published_at, start_date, end_date):
            continue
        description = item.get("description")
        category = _classify_article(f"{title}\n{description or ''}", ticker, info)
        articles.append(
            NewsArticle(
                title=title,
                source=str((item.get("source") or {}).get("name") or "GNews"),
                published_at=published_at,
                url=url,
                description=description,
                category=category,
            )
        )
    return _dedupe_articles(articles)[:limit]


def fetch_exa_news(
    query: str,
    *,
    ticker: str,
    info: dict | None,
    start_date: date,
    end_date: date,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return []

    payload = {
        "query": query,
        "numResults": min(100, max(1, limit)),
        "startPublishedDate": f"{start_date.isoformat()}T00:00:00.000Z",
        "endPublishedDate": f"{end_date.isoformat()}T23:59:59.999Z",
        "type": "keyword",
        "useAutoprompt": True,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.post(EXA_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    articles: list[NewsArticle] = []
    for item in data.get("results", []):
        title = item.get("title")
        url = item.get("url")
        if not title or not url:
            continue
        published_at = _to_datetime(item.get("publishedDate"))
        if not _in_date_range(published_at, start_date, end_date):
            continue
        description = item.get("text")
        category = _classify_article(f"{title}\n{description or ''}", ticker, info)
        articles.append(
            NewsArticle(
                title=title,
                source="Exa AI",
                published_at=published_at,
                url=url,
                description=description,
                category=category,
            )
        )
    return _dedupe_articles(articles)[:limit]


def fetch_relevant_news(
    ticker: str,
    start_date: date,
    end_date: date,
    limit: int,
    timeout_seconds: float,
) -> list[NewsArticle]:
    # Be robust: each source failure is isolated and never fails the whole request.
    pool_size = min(80, max(limit * settings.news_fetch_pool_multiplier, limit))
    articles: list[NewsArticle] = []
    ticker_info: dict | None = None
    try:
        ticker_obj = yf.Ticker(ticker)
        ticker_info = ticker_obj.info if isinstance(ticker_obj.info, dict) else {}
    except Exception:
        ticker_info = None

    competitors = top_competitors_for_news_search(ticker, ticker_info)
    base_terms = [ticker, "stock", "earnings", "industry", "macroeconomy", "regulation"]
    general_query = " ".join(base_terms + competitors)

    workers = max(1, min(settings.news_fetch_parallel_workers, 5))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_name: dict = {
            pool.submit(
                fetch_company_news_yfinance,
                ticker,
                start_date,
                end_date,
                pool_size,
            ): "yfinance",
            pool.submit(
                fetch_newsapi_news,
                general_query,
                ticker=ticker,
                info=ticker_info,
                start_date=start_date,
                end_date=end_date,
                limit=pool_size,
                timeout_seconds=timeout_seconds,
            ): "newsapi",
            pool.submit(
                fetch_gnews_news,
                general_query,
                ticker=ticker,
                info=ticker_info,
                start_date=start_date,
                end_date=end_date,
                limit=pool_size,
                timeout_seconds=timeout_seconds,
            ): "gnews",
            pool.submit(
                fetch_exa_news,
                general_query,
                ticker=ticker,
                info=ticker_info,
                start_date=start_date,
                end_date=end_date,
                limit=pool_size,
                timeout_seconds=timeout_seconds,
            ): "exa",
            pool.submit(
                fetch_general_news_jina,
                general_query,
                pool_size,
                timeout_seconds,
            ): "jina",
        }
        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                fetched = fut.result()
                articles.extend(fetched)
            except Exception as exc:
                logger.warning("News provider %s failed for %s: %s", name, ticker, exc)

    return _finalize_news_with_relevance(
        articles,
        ticker,
        ticker_info,
        min_score=settings.news_relevance_threshold,
        limit=limit,
    )
