from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


MovementType = Literal["up", "down", "all"]


class StockDay(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    pct_change: float
    is_major_move: bool


class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: datetime | None = None
    url: str
    description: str | None = None
    category: Literal["company", "competitor", "industry", "macro", "unknown"] = "unknown"
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Post-fetch relevance to the ticker (0..1).")


class MovementWithNews(BaseModel):
    stock_day: StockDay
    related_news: list[NewsArticle]


class StockNewsResponse(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    threshold_pct: float
    movement_type: MovementType
    days: list[StockDay]
    major_movements: list[MovementWithNews]


class ChatRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15)
    question: str = Field(..., min_length=3, max_length=1000)
    start_date: date | None = None
    end_date: date | None = None
    threshold_pct: float = 2.0
    movement_type: MovementType = "all"
    news_limit: int = Field(10, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    highlights: list[str]
