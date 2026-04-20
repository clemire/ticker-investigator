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
# Edit .env: add API keys you plan to use (see below). Per-article `news_drop_*` lines log at INFO.
uvicorn app.main:app --reload
```

`LOG_LEVEL` in `.env` is applied when the app module loads. You can mirror it for uvicorn’s own logs with e.g. `uvicorn app.main:app --log-level debug`.

API docs: `http://127.0.0.1:8000/docs`

## Environment (`.env`)

The app loads `.env` automatically (`python-dotenv` via `app/config.py`).

| Variable | Purpose |
|----------|---------|
| `LOG_LEVEL` | Root log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default `INFO`). `news_drop_relevance` / `news_drop_limit` per-article lines use `INFO`. |
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

## CLI client (`chat_cli.py`)

Interactive terminal client: it loads **`GET /api/v1/stock-news`** from your running API, builds a compact context for the model, and sends your questions to an **OpenAI-compatible** chat endpoint (`/v1/chat/completions`).

### Prerequisites

1. **Ticker Investigator API** is running (see [Quick start](#quick-start)), default `http://127.0.0.1:8000`.
2. **`OPENAI_API_KEY`** (or the env name you pass with `--llm-api-key-env`) is set for the LLM call.

### Start the client

```bash
# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — CLI (key from .env or export)
python chat_cli.py --ticker AAPL
```

Narrow the window or tune news volume:

```bash
python chat_cli.py --ticker MSFT \
  --start-date 2025-01-01 --end-date 2025-03-31 \
  --threshold-pct 2.5 --news-limit 15
```

Point at another API host or model:

```bash
python chat_cli.py --ticker NVDA \
  --api-base-url http://127.0.0.1:8000 \
  --llm-base-url https://api.openai.com/v1 \
  --llm-model gpt-4o-mini
```

Useful **flags** (see `python chat_cli.py --help` for all): `--start-date`, `--end-date`, `--threshold-pct`, `--movement-type` (`up` | `down` | `all`), `--news-limit`, `--api-base-url`, `--llm-base-url`, `--llm-model`, `--timeout`, `--api-retries`.

While the initial dataset loads, a short **spinner** line is written to **stderr** (TTY only).

### REPL commands

These are handled locally (no LLM call):

| Command | Action |
|--------|--------|
| `:help` | Show built-in commands |
| `:quit` | Exit (also **Ctrl+D** / **Ctrl+C**) |
| `:reload` | Refetch `stock-news` with the same CLI flags and reset chat history |
| `:list` | Print every **major price movement**, numbered **1…N** in order of **largest \|% change\|** first |
| `:list N` | Print **all related articles** for movement **#N** (same numbering as `:list`), with title, category, source, published time, and URL |

Example session:

```text
ask> :list
→ prints every major-move day, numbered 1…N (largest |% change| first)

ask> :list 1
→ prints all linked articles for movement #1 (title, category, source, URL, …)

ask> What themes show up in the news on the biggest down days?
→ model answer using only the loaded dataset

ask> :reload
→ refetches stock/news; chat history resets to a fresh system prompt

ask> :quit
```

Natural-language questions use the loaded dataset only; the model does not browse the web.

### Environment

The CLI reads **`OPENAI_API_KEY`** from the environment (default; override with `--llm-api-key-env`). If you use a **`.env`** file for the API process, export the same variables in the shell that runs `chat_cli.py`, or load them with your shell or a tool like `direnv`.

## Notes

- **Relevance** blends direct ticker/company overlap with industry/macro keyword matches (aligned with classification). Tune `NEWS_RELEVANCE_THRESHOLD` if you see too many drops or too much noise (raising it above ~0.45 can drop thematic-only articles again).
- **Classification** uses the LLM when enabled; otherwise the same keyword rules as before. Results are cached in-process by URL+title hash to avoid repeat API cost.
- Price history can occasionally be empty from `yfinance`; the API retries before returning 404.

# Quickstart

1. Start the server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add API key for OPENAI
# (Only OPENAI is required.)
uvicorn app.main:app --reload
```

2. Run the cli client

```bash
python chat_cli.py --ticker AAPL
```

You can optionally specify start and end dates, as well as many other parameters. The app will default to a 4 month period ending today. Run `python chat_cli.py` to see a full list of params.

## Example client output

```
Commands: :quit — exit | :reload — refetch dataset | :list — all major moves | :list N — articles for move #N (same order as :list) | :help — commands

ask> :list

Major price movements for GOOG (2026-01-20 to 2026-04-20), 14 day(s) at or above threshold:

1. **2026-03-31**: Close at $286.86, pct change +5.02%
2. **2026-02-20**: Close at $314.68, pct change +3.74%
3. **2026-04-14**: Close at $330.58, pct change +3.56%
4. **2026-04-08**: Close at $314.74, pct change +3.56%
5. **2026-03-24**: Close at $289.20, pct change -3.28%
6. **2026-03-26**: Close at $280.74, pct change -3.06%
7. **2026-04-01**: Close at $294.90, pct change +2.80%
8. **2026-03-09**: Close at $306.01, pct change +2.66%
9. **2026-03-27**: Close at $273.76, pct change -2.49%
10. **2026-02-06**: Close at $322.87, pct change -2.48%
11. **2026-02-11**: Close at $311.11, pct change -2.29%
12. **2026-03-20**: Close at $298.79, pct change -2.27%
13. **2026-02-04**: Close at $333.11, pct change -2.16%
14. **2026-04-07**: Close at $303.93, pct change +2.11%

ask> :list 1

**Movement 1** (GOOG) — **2026-03-31** — close $286.86, pct change +5.02%

Articles (7):

1. **Alphabet Inc. Class C Stock: Core Business Drivers, Investor ...**
   - **Category**: company | **Source**: Exa AI | **Published**: 2026-04-01T00:00:00Z
   - https://www.ad-hoc-news.de/boerse/ueberblick/alphabet-inc-class-c-stock-core-business-drivers-investor-relevance/69047846
2. **Alphabet Shares Climb as Analysts Affirm Long-Term Buy… - Inkl**
   - **Category**: company | **Source**: Exa AI | **Published**: 2026-03-31T00:00:00Z
   - https://www.inkl.com/news/alphabet-shares-climb-as-analysts-affirm-long-term-buy-on-ai-momentum-and-cloud-surge
3. **Alphabet's Solution Is A Gift For Micron's Memory Sales (NASDAQ:MU)**
   - **Category**: competitor | **Source**: Exa AI | **Published**: 2026-03-30T00:00:00Z
   - https://seekingalpha.com/article/4887033-alphabets-solution-is-a-gift-for-microns-memory-sales
4. **Shares of Alphabet fell to their lowest level since late 2025, dropping ...**
   - **Category**: company | **Source**: Exa AI | **Published**: 2026-03-31T00:00:00Z
   - https://www.facebook.com/headingsus/posts/shares-of-alphabet-fell-to-their-lowest-level-since-late-2025-dropping-sharply-e/936981202663575/
5. **how have the themes discussed in Google's earnings calls changed ...**
   - **Category**: company | **Source**: Exa AI | **Published**: 2026-03-30T00:00:00Z
   - https://www.factiq.com/share/d5ec9046e9b947259be00cdcfba60b80
6. **E-Commerce Sector Update – April 2026 - Capstone Partners**
   - **Category**: industry | **Source**: Exa AI | **Published**: 2026-04-01T00:00:00Z
   - https://www.capstonepartners.com/insights/article-e-commerce-sector-update/
7. **Meta faces regulatory risks but Jefferies sees buying opportunity**
   - **Category**: competitor | **Source**: Exa AI | **Published**: 2026-03-30T00:00:00Z
   - https://finance.yahoo.com/markets/stocks/articles/meta-faces-regulatory-risks-jefferie-193100680.html

ask> ...
```

# Comments

## Am I happy with my solution?

Not quite. I use chatgpt to support chat with the terminal client util that allows you to interact with articles. I find that it doesn’t consistently render data and sometimes omits information. I think this is not very usable. Using keywords to include industry and macro articles also seems a bit brittle and limits search results; however it does enrich the returned response.

## Process

I used AI to build out a FastAPI service that searches for price movements, then finds articles from supported sources in parallel using a keyword search and a timeline of +-1 day from the price movement. 

We use several keyword queries for each news provider: one for the company itself, and one for each of the top 5 competitors as determined by an llm query. Articles are deduped, assigned a category (“company/industry/macro/unknown”) and given a relevance score on title+description. Articles below the relevance threshold are dropped. Once articles are dropped, the remaining articles are optionally reclassified with an llm for more accurate categories.

I considered adding UX, but opted to build a text client since I needed to support a chat interface anyway, and also because I’m a backend engineer and didn’t want to get bogged down in UI bugs. 

## What would I do differently?

I think I would spend more time thinking about how to surface articles that are relevant to the macro category. I would also tighten the client interface by specifying in a prompt how I want results to render, for the sake of consistency and discoverability of data.

## Did I get stuck anywhere?

I got a little caught up in why yfinance wasn’t returning articles, and whether not getting (many) articles back was due to a bug in my code or because my api keys had run out of capacity. Thankfully my exa key renewed after a few minutes and I was able to roughly validate how I process articles and confirm categories were sane. If exa hadn’t renewed capacity, I would have needed a different news source.