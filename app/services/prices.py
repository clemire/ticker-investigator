from datetime import date, timedelta
import logging
import time

import yfinance as yf

from app.config import settings
from app.models import StockDay

logger = logging.getLogger(__name__)


def _safe_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


def fetch_stock_days(ticker: str, start_date: date, end_date: date) -> list[StockDay]:
    # yfinance end date is exclusive, so add one day.
    y_end_date = end_date + timedelta(days=1)
    history = None
    for attempt in range(1, settings.upstream_retry_attempts + 1):
        try:
            history = yf.Ticker(ticker).history(start=start_date.isoformat(), end=y_end_date.isoformat())
        except Exception as exc:
            logger.warning(
                "Price history fetch failed for %s on attempt %s/%s: %s",
                ticker,
                attempt,
                settings.upstream_retry_attempts,
                exc,
            )
            history = None

        if history is not None and not history.empty:
            break

        if attempt < settings.upstream_retry_attempts:
            logger.warning(
                "Empty price history for %s on attempt %s/%s; retrying",
                ticker,
                attempt,
                settings.upstream_retry_attempts,
            )
            time.sleep(settings.upstream_retry_backoff_seconds * attempt)

    if history is None or history.empty:
        return []

    days: list[StockDay] = []
    prev_close: float | None = None
    for ts, row in history.iterrows():
        close = _safe_float(row.get("Close"))
        pct_change = 0.0
        if prev_close and prev_close != 0:
            pct_change = ((close - prev_close) / prev_close) * 100.0

        day = StockDay(
            date=ts.date(),
            open=_safe_float(row.get("Open")),
            high=_safe_float(row.get("High")),
            low=_safe_float(row.get("Low")),
            close=close,
            volume=_safe_int(row.get("Volume")),
            pct_change=round(pct_change, 4),
            is_major_move=False,
        )
        days.append(day)
        prev_close = close

    return days


def mark_major_moves(days: list[StockDay], threshold_pct: float) -> list[StockDay]:
    marked_days: list[StockDay] = []
    for day in days:
        is_major = abs(day.pct_change) >= threshold_pct
        marked_days.append(day.model_copy(update={"is_major_move": is_major}))
    return marked_days
