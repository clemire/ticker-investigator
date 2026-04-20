# Ticker Investigator

Mini app that explains major stock price movements using relevant news.

## Features

- Fetches historical stock prices by ticker (via `yfinance`)
- Detects major daily movements (default: absolute daily move >= 2%)
- Pulls company and broader news around movement windows
- Exposes:
  - `GET /api/v1/stock-news` for stock + movement + related news data
  - `POST /api/v1/chat` for Q&A against the computed dataset

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open docs at: `http://127.0.0.1:8000/docs`

### Environment config (`.env`)

The app auto-loads environment variables from `.env`.

```bash
cp .env.example .env
```

Then fill keys in `.env`:

- `NEWSAPI_API_KEY`
- `GNEWS_API_KEY`
- `EXA_API_KEY`
- `OPENAI_API_KEY` (for `chat_cli.py`)
- Optional resilience tuning:
  - `UPSTREAM_RETRY_ATTEMPTS`
  - `UPSTREAM_RETRY_BACKOFF_SECONDS`

## API

### `GET /api/v1/stock-news`

Query params:

- `ticker` (required), e.g. `AAPL`
- `start_date` (optional, `YYYY-MM-DD`, default `end_date-90d`)
- `end_date` (optional, default `today`)
- `threshold_pct` (optional, default `2.0`)
- `movement_type` (optional: `up`, `down`, `all`; default `all`)
- `news_limit` (optional, default `10`, max `50`)

Example:

```bash
curl "http://127.0.0.1:8000/api/v1/stock-news?ticker=AAPL&threshold_pct=2&movement_type=all&news_limit=8"
```

### `POST /api/v1/chat`

Body:

```json
{
  "ticker": "AAPL",
  "question": "Why did the stock move the most in this period?",
  "start_date": "2025-01-01",
  "end_date": "2025-03-31",
  "threshold_pct": 2.0,
  "movement_type": "all",
  "news_limit": 10
}
```

Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "question": "What was the biggest move and why?"
  }'
```

## Notes

- News coverage quality depends on external provider availability.
- The chat endpoint is retrieval-style and deterministic (no paid LLM required).
- News collection now uses these providers: `NewsAPI`, `GNews`, `Jina AI`, `Exa AI`.
- Optional API keys (recommended): `NEWSAPI_API_KEY`, `GNEWS_API_KEY`, `EXA_API_KEY`.
- If a provider fails or has no key configured, the API continues with remaining sources.

## Terminal LLM Chat

Use the terminal client when you want arbitrary natural-language investigation over a ticker dataset.

1) Start the API:

```bash
uvicorn app.main:app --reload
```

2) Set `OPENAI_API_KEY` in `.env` (or export it manually).

3) Start interactive chat:

```bash
python chat_cli.py --ticker AAPL
```

Optional flags:

- `--start-date 2025-01-01 --end-date 2025-03-31`
- `--threshold-pct 2.0 --movement-type all --news-limit 10`
- `--llm-model gpt-4o-mini`
- `--llm-base-url https://api.openai.com/v1` (or any OpenAI-compatible endpoint)
- `--llm-api-key-env OPENAI_API_KEY`

In-session commands:

- `:help` show commands
- `:reload` refresh ticker data from API
- `:quit` exit
