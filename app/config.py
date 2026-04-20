import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    default_threshold_pct: float = float(os.getenv("DEFAULT_THRESHOLD_PCT", "2.0"))
    default_news_limit: int = int(os.getenv("DEFAULT_NEWS_LIMIT", "10"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15.0"))
    upstream_retry_attempts: int = int(os.getenv("UPSTREAM_RETRY_ATTEMPTS", "3"))
    upstream_retry_backoff_seconds: float = float(os.getenv("UPSTREAM_RETRY_BACKOFF_SECONDS", "1.0"))
    # Post-fetch: keep articles with relevance_score >= threshold (0..1).
    news_relevance_threshold: float = float(os.getenv("NEWS_RELEVANCE_THRESHOLD", "0.42"))
    # Fetch more raw articles per provider before filtering so enough pass relevance.
    news_fetch_pool_multiplier: int = int(os.getenv("NEWS_FETCH_POOL_MULTIPLIER", "4"))
    # Concurrent news provider HTTP calls (yfinance + NewsAPI + GNews + Exa + Jina).
    news_fetch_parallel_workers: int = int(os.getenv("NEWS_FETCH_PARALLEL", "5"))
    # Batched LLM classification (see app/services/llm_article_classifier.py)
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    news_llm_model: str = os.getenv("NEWS_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    news_llm_batch_size: int = int(os.getenv("NEWS_LLM_BATCH_SIZE", "12"))
    news_llm_timeout_seconds: float = float(os.getenv("NEWS_LLM_TIMEOUT_SECONDS", "45.0"))
    news_llm_max_parallel_chunks: int = int(os.getenv("NEWS_LLM_MAX_PARALLEL", "3"))


settings = Settings()
