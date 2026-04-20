from datetime import date, timedelta
import logging

from fastapi import FastAPI, HTTPException, Query

from app.config import settings
from app.models import ChatRequest, ChatResponse, MovementType, StockNewsResponse
from app.services.analyzer import build_stock_news_response
from app.services.chat import answer_question

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ticker Investigator API",
    description="Explain major stock moves with relevant news.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/stock-news", response_model=StockNewsResponse)
def get_stock_and_news(
    ticker: str = Query(..., min_length=1, max_length=15, description="Public ticker like AAPL"),
    start_date: date | None = Query(None, description="Inclusive start date (YYYY-MM-DD)"),
    end_date: date | None = Query(None, description="Inclusive end date (YYYY-MM-DD)"),
    threshold_pct: float = Query(settings.default_threshold_pct, ge=0.1, le=50.0),
    movement_type: MovementType = Query("all"),
    news_limit: int = Query(settings.default_news_limit, ge=1, le=50),
) -> StockNewsResponse:
    actual_end = end_date or date.today()
    actual_start = start_date or (actual_end - timedelta(days=90))
    if actual_start > actual_end:
        raise HTTPException(status_code=400, detail="start_date must be before or equal to end_date")

    try:
        response = build_stock_news_response(
            ticker=ticker.strip().upper(),
            start_date=actual_start,
            end_date=actual_end,
            threshold_pct=threshold_pct,
            movement_type=movement_type,
            news_limit=news_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch stock/news data: {exc}") from exc

    if not response.days:
        logger.warning(
            "No price history found for ticker=%s start_date=%s end_date=%s threshold_pct=%s movement_type=%s",
            ticker,
            actual_start,
            actual_end,
            threshold_pct,
            movement_type,
        )
        raise HTTPException(status_code=404, detail=f"No price history found for ticker '{ticker}'")

    return response


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_with_stock_data(payload: ChatRequest) -> ChatResponse:
    actual_end = payload.end_date or date.today()
    actual_start = payload.start_date or (actual_end - timedelta(days=90))
    if actual_start > actual_end:
        raise HTTPException(status_code=400, detail="start_date must be before or equal to end_date")

    try:
        data = build_stock_news_response(
            ticker=payload.ticker.strip().upper(),
            start_date=actual_start,
            end_date=actual_end,
            threshold_pct=payload.threshold_pct,
            movement_type=payload.movement_type,
            news_limit=payload.news_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch stock/news data: {exc}") from exc

    if not data.days:
        logger.warning(
            "No price history found for chat ticker=%s start_date=%s end_date=%s threshold_pct=%s movement_type=%s",
            payload.ticker,
            actual_start,
            actual_end,
            payload.threshold_pct,
            payload.movement_type,
        )
        raise HTTPException(status_code=404, detail=f"No price history found for ticker '{payload.ticker}'")

    return answer_question(payload.question, data)
