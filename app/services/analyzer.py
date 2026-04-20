from datetime import date, timedelta

from app.config import settings
from app.models import MovementType, MovementWithNews, StockNewsResponse
from app.services.news import fetch_relevant_news
from app.services.prices import fetch_stock_days, mark_major_moves


def _filter_movement_type(pct_change: float, movement_type: MovementType) -> bool:
    if movement_type == "all":
        return True
    if movement_type == "up":
        return pct_change > 0
    return pct_change < 0


def build_stock_news_response(
    ticker: str,
    start_date: date,
    end_date: date,
    threshold_pct: float,
    movement_type: MovementType = "all",
    news_limit: int = 10,
) -> StockNewsResponse:
    days = fetch_stock_days(ticker=ticker, start_date=start_date, end_date=end_date)
    marked_days = mark_major_moves(days=days, threshold_pct=threshold_pct)

    major_days = [d for d in marked_days if d.is_major_move and _filter_movement_type(d.pct_change, movement_type)]
    movements: list[MovementWithNews] = []
    for day in major_days:
        # Keep a narrow event window around the movement date.
        window_start = day.date - timedelta(days=1)
        window_end = day.date + timedelta(days=1)
        try:
            news = fetch_relevant_news(
                ticker=ticker,
                start_date=window_start,
                end_date=window_end,
                limit=news_limit,
                timeout_seconds=settings.request_timeout_seconds,
            )
        except Exception:
            # Stock movement data should still be returned if all news providers fail.
            news = []
        movements.append(MovementWithNews(stock_day=day, related_news=news))

    return StockNewsResponse(
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=end_date,
        threshold_pct=threshold_pct,
        movement_type=movement_type,
        days=marked_days,
        major_movements=movements,
    )
