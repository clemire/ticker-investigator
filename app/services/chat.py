from collections import Counter

from app.models import ChatResponse, MovementWithNews, StockNewsResponse


def _movement_sentence(movement: MovementWithNews) -> str:
    day = movement.stock_day
    direction = "rose" if day.pct_change > 0 else "fell"
    return f"{day.date.isoformat()}: {direction} {abs(day.pct_change):.2f}% (close {day.close:.2f})"


def _categorize_counts(data: StockNewsResponse) -> Counter:
    counts: Counter = Counter()
    for m in data.major_movements:
        for article in m.related_news:
            counts[article.category] += 1
    return counts


def answer_question(question: str, data: StockNewsResponse) -> ChatResponse:
    q = question.lower()
    highlights: list[str] = []
    if not data.major_movements:
        return ChatResponse(
            answer=f"No major moves found for {data.ticker} in this date range at {data.threshold_pct:.2f}% threshold.",
            highlights=[],
        )

    top_moves = sorted(data.major_movements, key=lambda m: abs(m.stock_day.pct_change), reverse=True)
    for move in top_moves[:5]:
        highlights.append(_movement_sentence(move))

    if any(term in q for term in ("biggest", "largest", "max", "most")):
        biggest = top_moves[0]
        answer = (
            f"The largest move was on {biggest.stock_day.date.isoformat()} at "
            f"{biggest.stock_day.pct_change:.2f}%. "
            f"I found {len(biggest.related_news)} related articles around that date."
        )
        return ChatResponse(answer=answer, highlights=highlights)

    if any(term in q for term in ("why", "reason", "cause", "because")):
        biggest = top_moves[0]
        first_titles = [a.title for a in biggest.related_news[:3]]
        if first_titles:
            joined_titles = "; ".join(first_titles)
            answer = (
                f"A likely driver for the largest move on {biggest.stock_day.date.isoformat()} "
                f"({biggest.stock_day.pct_change:.2f}%) is reflected in these headlines: {joined_titles}."
            )
        else:
            answer = (
                f"I found no near-date headlines for the largest move on {biggest.stock_day.date.isoformat()}, "
                "so the move may be technical or driven by broader market factors."
            )
        return ChatResponse(answer=answer, highlights=highlights)

    if any(term in q for term in ("macro", "industry", "company", "competitor", "category", "theme")):
        counts = _categorize_counts(data)
        answer = (
            f"News mix for major-move windows: company={counts.get('company', 0)}, "
            f"competitor={counts.get('competitor', 0)}, "
            f"industry={counts.get('industry', 0)}, macro={counts.get('macro', 0)}, "
            f"unknown={counts.get('unknown', 0)}."
        )
        return ChatResponse(answer=answer, highlights=highlights)

    answer = (
        f"I found {len(data.major_movements)} major moves for {data.ticker}. "
        f"Top move: {top_moves[0].stock_day.date.isoformat()} ({top_moves[0].stock_day.pct_change:.2f}%). "
        "Ask about biggest move, likely reasons, or category mix for more detail."
    )
    return ChatResponse(answer=answer, highlights=highlights)
