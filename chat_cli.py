#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
import time
from datetime import date, timedelta
from typing import Any

import httpx

_CLEAR_LINE = "\033[2K\r"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive LLM chat over ticker-investigator stock/news data.",
    )
    parser.add_argument("--ticker", required=True, help="Public ticker, e.g. AAPL")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000", help="Ticker Investigator API base URL")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--threshold-pct", type=float, default=2.0, help="Major move threshold percentage")
    parser.add_argument(
        "--movement-type",
        default="all",
        choices=["up", "down", "all"],
        help="Major movement direction filter",
    )
    parser.add_argument("--news-limit", type=int, default=10, help="Related news limit per movement")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--llm-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model name (OpenAI-compatible)",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing LLM API key",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-retries", type=int, default=3, help="Retries when loading ticker dataset")
    return parser.parse_args()


def resolve_dates(start_date_str: str | None, end_date_str: str | None) -> tuple[str, str]:
    today = date.today()
    end_date = date.fromisoformat(end_date_str) if end_date_str else today
    start_date = date.fromisoformat(start_date_str) if start_date_str else (end_date - timedelta(days=90))
    return start_date.isoformat(), end_date.isoformat()


def fetch_stock_news_dataset(args: argparse.Namespace) -> dict[str, Any]:
    start_date, end_date = resolve_dates(args.start_date, args.end_date)
    params = {
        "ticker": args.ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "threshold_pct": args.threshold_pct,
        "movement_type": args.movement_type,
        "news_limit": args.news_limit,
    }
    url = f"{args.api_base_url.rstrip('/')}/api/v1/stock-news"
    last_error: Exception | None = None
    for attempt in range(1, args.api_retries + 1):
        try:
            with httpx.Client(timeout=args.timeout) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"API returned 404 for {args.ticker.upper()} in {start_date}..{end_date}. "
                        f"Server detail: {resp.text}"
                    )
                resp.raise_for_status()
                return resp.json()
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, RuntimeError) as exc:
            last_error = exc
            if attempt >= args.api_retries:
                break
            print(f"Dataset load attempt {attempt}/{args.api_retries} failed; retrying...", file=sys.stderr)
            time.sleep(min(2 * attempt, 5))

    if last_error is None:
        raise RuntimeError("Ticker dataset fetch failed for an unknown reason")
    raise last_error


def load_dataset_with_spinner(args: argparse.Namespace) -> dict[str, Any]:
    """Run the initial API fetch with a stderr spinner and short blurb."""
    start_date, end_date = resolve_dates(args.start_date, args.end_date)
    ticker_u = args.ticker.upper()
    if not sys.stderr.isatty():
        return fetch_stock_news_dataset(args)

    result: dict[str, Any] | None = None
    error: Exception | None = None

    def worker() -> None:
        nonlocal result, error
        try:
            result = fetch_stock_news_dataset(args)
        except Exception as exc:
            error = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    blurb = (
        f"Fetching prices and searching news for {ticker_u} "
        f"({start_date} → {end_date}) — NewsAPI, GNews, Exa, Jina…"
    )
    i = 0
    while thread.is_alive():
        ch = frames[i % len(frames)]
        sys.stderr.write(f"{_CLEAR_LINE}{ch} {blurb}")
        sys.stderr.flush()
        time.sleep(0.09)
        i += 1
    thread.join()
    sys.stderr.write(_CLEAR_LINE)
    sys.stderr.flush()
    if error is not None:
        raise error
    if result is None:
        raise RuntimeError("Ticker dataset load finished without result")
    return result


def compact_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    days = dataset.get("days", [])
    major = dataset.get("major_movements", [])
    # Include every major move so the model matches num_major_movements (do not cap here).
    top_major = sorted(major, key=lambda m: abs(m["stock_day"]["pct_change"]), reverse=True)

    compact_major: list[dict[str, Any]] = []
    for movement in top_major:
        stock_day = movement["stock_day"]
        related_news = movement.get("related_news", [])[:8]
        compact_major.append(
            {
                "date": stock_day["date"],
                "close": stock_day["close"],
                "pct_change": stock_day["pct_change"],
                "volume": stock_day["volume"],
                "news": [
                    {
                        "title": article.get("title"),
                        "source": article.get("source"),
                        "published_at": article.get("published_at"),
                        "category": article.get("category"),
                        "relevance_score": article.get("relevance_score"),
                        "url": article.get("url"),
                    }
                    for article in related_news
                ],
            }
        )

    return {
        "ticker": dataset.get("ticker"),
        "date_range": [dataset.get("start_date"), dataset.get("end_date")],
        "threshold_pct": dataset.get("threshold_pct"),
        "movement_type": dataset.get("movement_type"),
        "num_days": len(days),
        "num_major_movements": len(major),
        "major_movements": compact_major,
    }


def build_system_prompt(dataset_context: dict[str, Any]) -> str:
    context_json = json.dumps(dataset_context, ensure_ascii=True)
    return (
        "You are a stock movement investigation assistant. "
        "Answer questions only from the provided ticker dataset context. "
        "If information is missing, say what is missing and ask for a refresh with broader filters. "
        "Be concise but analytical. Quote dates, percentages, and relevant headlines when possible. "
        "When listing major price movements, include every entry in major_movements exactly once "
        "(len(major_movements) must equal num_major_movements); do not omit the smallest moves.\n\n"
        f"Ticker dataset context JSON:\n{context_json}"
    )


def wants_full_movement_enumeration(question: str) -> bool:
    """Catalogue-style questions where we must list every move (LLMs often drop one otherwise)."""
    ql = question.lower()
    if "movement" not in ql and "moves" not in ql:
        return False
    if any(
        x in ql
        for x in (
            "why ",
            "what caused",
            "reason for",
            "because ",
            "explain ",
            "cause of",
        )
    ):
        return False
    catalogue = (
        "all major",
        "all price",
        "every major",
        "every price",
        "list all",
        "list the",
        "list major",
        "list every",
        "show all",
        "show me the price",
        "show the price",
        "complete list",
        "full list",
        "all movements",
        "all moves",
        "price movements",
        "major movements",
        "enumerate",
        "what are the",
        "what were the",
        "give me the",
    )
    return any(c in ql for c in catalogue)


def format_major_movements_answer(dataset: dict[str, Any]) -> str:
    """Deterministic full list: same ordering as compact_dataset (|pct_change| desc)."""
    major = dataset.get("major_movements", [])
    if not major:
        return "No major movements in this dataset."
    ordered = sorted(major, key=lambda m: abs(m["stock_day"]["pct_change"]), reverse=True)
    lines: list[str] = []
    for i, m in enumerate(ordered, start=1):
        d = m["stock_day"]
        date_s = d["date"]
        lines.append(
            f"{i}. **{date_s}**: Close at ${float(d['close']):.2f}, pct change {float(d['pct_change']):+.2f}%"
        )
    ticker = dataset.get("ticker", "")
    start_d = dataset.get("start_date")
    end_d = dataset.get("end_date")
    head = (
        f"The major price movements for {ticker} from {start_d} to {end_d} "
        f"({len(ordered)} days at or above the threshold):\n\n"
    )
    return head + "\n".join(lines)


def llm_chat_completion(
    *,
    llm_base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> str:
    url = f"{llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc


def print_help() -> None:
    print("Commands: :quit to exit, :reload to refresh ticker dataset from API, :help for commands")


def main() -> int:
    args = parse_args()
    api_key = os.getenv(args.llm_api_key_env)
    if not api_key:
        print(f"Missing API key env var: {args.llm_api_key_env}", file=sys.stderr)
        return 1

    try:
        dataset = load_dataset_with_spinner(args)
    except Exception as exc:
        print(f"Failed to fetch ticker dataset: {exc}", file=sys.stderr)
        return 1

    dataset_context = compact_dataset(dataset)
    system_prompt = build_system_prompt(dataset_context)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print(f"Loaded {dataset_context['ticker']} dataset with {dataset_context['num_major_movements']} major movements.")
    print_help()

    while True:
        try:
            user_input = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_input:
            continue
        if user_input == ":quit":
            print("Exiting.")
            return 0
        if user_input == ":help":
            print_help()
            continue
        if user_input == ":reload":
            try:
                dataset = load_dataset_with_spinner(args)
                dataset_context = compact_dataset(dataset)
                system_prompt = build_system_prompt(dataset_context)
                messages = [{"role": "system", "content": system_prompt}]
                print(
                    f"Reloaded {dataset_context['ticker']} dataset with "
                    f"{dataset_context['num_major_movements']} major movements."
                )
            except Exception as exc:
                print(f"Reload failed: {exc}", file=sys.stderr)
            continue

        if wants_full_movement_enumeration(user_input):
            answer = format_major_movements_answer(dataset)
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": answer})
            print(f"\n{answer.strip()}")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            answer = llm_chat_completion(
                llm_base_url=args.llm_base_url,
                model=args.llm_model,
                api_key=api_key,
                messages=messages,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"LLM request failed: {exc}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"\n{answer.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
