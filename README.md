# Ticker Investigator

Mini app that ties **historical stock moves** to **news**, with a small HTTP API and an optional terminal LLM client.

## What it does

- **Prices**: Daily OHLCV by ticker via `yfinance`, with day-over-day % change and configurable “major move” days (default: absolute move ≥ **2%**).
- **News**: For each major-move day, searches a **±1 day** window and aggregates articles from several backends **in parallel** (thread pool; size `NEWS_FETCH_PARALLEL`, max 5):
  - **Yahoo Finance** headlines (`yfinance` ticker news)
  - **NewsAPI** (`everything` search) — requires `NEWSAPI_API_KEY`
  - **GNews** — requires `GNEWS_API_KEY`
  - **Exa AI** — requires `EXA_API_KEY`
  - **Jina AI** (wraps Google News RSS) — no key; used as a fallback
- **Post-fetch**: Each article gets a **relevance score** (0–1) from (1) ticker/company-name overlap in the text and (2) the same **industry** and **macro** keyword lists used for tagging—so sector- or policy-heavy pieces without the symbol in the headline are not all discarded. Items below `NEWS_RELEVANCE_THRESHOLD` are **dropped**. Results are sorted by score, then capped by `news_limit`.
- **Tags** (`company` / `industry` / `macro` / `unknown`): After relevance filtering, articles are **re-labeled by a batched LLM** (one JSON response per chunk of up to `NEWS_LLM_BATCH_SIZE` items; chunks can run in parallel). `OPENAI_API_KEY` must be set; set `NEWS_LLM_CLASSIFIER_ENABLED=false` to skip and keep fast keyword labels only. Keyword heuristics still apply when the LLM is off or a chunk fails.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add API keys you plan to use (see below).
uvicorn app.main:app --reload
```

API docs: `http://127.0.0.1:8000/docs`

## Environment (`.env`)

The app loads `.env` automatically (`python-dotenv` via `app/config.py`).

| Variable | Purpose |
|----------|---------|
| `DEFAULT_THRESHOLD_PCT` | Default major-move threshold (default `2.0`) |
| `DEFAULT_NEWS_LIMIT` | Default max related articles per move (default `10`) |
| `REQUEST_TIMEOUT_SECONDS` | HTTP timeout for upstream news calls |
| `UPSTREAM_RETRY_ATTEMPTS` | Retries when `yfinance` price history is empty |
| `UPSTREAM_RETRY_BACKOFF_SECONDS` | Backoff between price retries |
| `NEWS_RELEVANCE_THRESHOLD` | Min relevance score 0–1 to keep an article (default `0.42`) |
| `NEWS_FETCH_POOL_MULTIPLIER` | Raw fetch size per source ≈ `news_limit × multiplier` before filtering (default `4`) |
| `NEWS_FETCH_PARALLEL` | Thread pool size for parallel provider calls (default `5`, capped at 5 sources) |
| `NEWSAPI_API_KEY` | NewsAPI |
| `GNEWS_API_KEY` | GNews |
| `EXA_API_KEY` | Exa |
| `OPENAI_API_KEY` | **Required** for LLM article classification (and for `chat_cli.py`) |
| `OPENAI_MODEL` | Default model name (CLI); classification uses `NEWS_LLM_MODEL` if set |
| `OPENAI_BASE_URL` | OpenAI-compatible API base (default official OpenAI) |
| `NEWS_LLM_CLASSIFIER_ENABLED` | `true`/`false`; default on when `OPENAI_API_KEY` is set |
| `NEWS_LLM_MODEL` | Model for batched classification (default `gpt-4o-mini`) |
| `NEWS_LLM_BATCH_SIZE` | Articles per LLM request (default `12`) |
| `NEWS_LLM_TIMEOUT_SECONDS` | Per-request timeout (default `45`) |
| `NEWS_LLM_MAX_PARALLEL` | Parallel chunk workers when a batch splits (default `3`) |

Missing keys are skipped; Jina and `yfinance` can still contribute. Provider failures are logged and do not fail the whole request.

## API

### `GET /api/v1/stock-news`

Returns daily rows, major-move days, and **`related_news`** per major day.

Query parameters:

- `ticker` (required)
- `start_date`, `end_date` (optional; default window is last **90** days ending `today`)
- `threshold_pct`, `movement_type` (`up` | `down` | `all`), `news_limit` (1–50)

Each news item includes at least: `title`, `source`, `url`, `published_at`, `description`, `category`, **`relevance_score`**.

### `POST /api/v1/chat`

Rule-based Q&A over the **same** stock+news payload built for the given ticker and dates (no external LLM). Useful for quick probes from `curl` or scripts.

### `GET /health`

Liveness check.

## Terminal LLM chat (`chat_cli.py`)

Uses an **OpenAI-compatible** chat API to answer questions against the JSON from `GET /api/v1/stock-news` (your API must be running).

```bash
export OPENAI_API_KEY=...   # or set in .env
python chat_cli.py --ticker AAPL
```

Common flags: `--start-date`, `--end-date`, `--threshold-pct`, `--movement-type`, `--news-limit`, `--api-base-url`, `--llm-model`, `--llm-base-url`, `--timeout`, `--api-retries`.

In the REPL: `:reload` refetches data; `:quit` exits. A short loading line appears on stderr while the dataset loads.

## Notes

- **Relevance** blends direct ticker/company overlap with industry/macro keyword matches (aligned with classification). Tune `NEWS_RELEVANCE_THRESHOLD` if you see too many drops or too much noise (raising it above ~0.45 can drop thematic-only articles again).
- **Classification** uses the LLM when enabled; otherwise the same keyword rules as before. Results are cached in-process by URL+title hash to avoid repeat API cost.
- Price history can occasionally be empty from `yfinance`; the API retries before returning 404.
