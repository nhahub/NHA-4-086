"""
SAIA backend — a thin API layer between the SAIA frontend and Databricks.

Why this exists:
  The frontend (sahm_finance_platform.html) runs in the user's browser. It can
  never safely hold a Databricks access token, and Databricks' own REST API
  does not support being called directly from arbitrary browser origins
  (no CORS). So this small server sits in between:

    Browser (SAIA frontend)  --calls-->  This API (FastAPI)  --calls-->  Databricks

  Only this server ever sees DATABRICKS_TOKEN. It reads all connection info
  from environment variables (.env), runs the SQL, and returns plain JSON.

  This version reads from the GOLD layer (finance_intelligence_hub.gold),
  which already has clean, analytics-ready columns:
    gold.dim_companies       -> one snapshot row per ticker
    gold.fact_stock_prices   -> full daily price history per ticker
    gold.news_sentiments     -> news articles per ticker, plus FinBERT
                                 positive/negative/neutral sentiment scores
                                 (this replaced the old gold.fact_news table).
                                 Already stored newest-first by published_at —
                                 none of the queries below re-sort it with an
                                 explicit ORDER BY, since sorting is the
                                 expensive part of every one of these calls
                                 and the table's physical order already gives
                                 us what we need for free.
    gold.stock_price_predictions -> one row per ticker with the latest
                                 ML-generated 30-day price prediction

Endpoints (used by the frontend's API_CONFIG):
  GET /api/companies
  GET /api/stock-prices?ticker=AAPL
  GET /api/news?ticker=AAPL      (ticker is optional — omit for all news)
  GET /api/news/search?q=...     (server-side free-text search over news)
  GET /api/predictions
  GET /api/watchlist?email=...
  POST /api/watchlist/toggle
  GET /api/saved-news?email=...
  POST /api/saved-news/toggle

Run it:
  1. pip install -r requirements.txt
  2. cp .env.example .env   and fill in your real Databricks values
  3. uvicorn app:app --reload --port 8000

  For local testing without Databricks at all, set MOCK_MODE=true in .env —
  the server will then serve data from the JSON files in test_data/ instead.
"""

import json
import os
import queue
import sqlite3
import threading
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---- MOCK MODE --------------------------------------------------------
# Set MOCK_MODE=true in .env (or the shell) to serve data straight from the
# JSON files in test_data/ instead of connecting to Databricks at all.
MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


def _load_json(filename: str):
    path = os.path.join(TEST_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if MOCK_MODE:
    # In mock mode we don't touch Databricks or its env vars at all.
    DATABRICKS_SERVER_HOSTNAME = DATABRICKS_HTTP_PATH = DATABRICKS_TOKEN = None
    CATALOG = os.environ.get("DATABRICKS_CATALOG", "finance_intelligence_hub")
    SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "gold")
else:
    from databricks import sql

    # ---- Databricks connection info (from .env — never hardcode these) ----
    DATABRICKS_SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]  # e.g. dbc-f88d29c6-4087.cloud.databricks.com
    DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_HTTP_PATH"]              # e.g. /sql/1.0/warehouses/xxxxxxxxxxxxxxxx
    DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]                      # personal access token / service principal token
    CATALOG = os.environ.get("DATABRICKS_CATALOG", "finance_intelligence_hub")
    SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "gold")

app = FastAPI(title="SAIA Backend")


# ---- FRONTEND (serve the SAIA HTML on the same Railway domain) --------
# The frontend is a single static HTML file. Instead of hosting it
# separately (e.g. GitHub Pages), we serve it directly from this same
# FastAPI app on "/", so the whole product lives at one Railway URL and
# there's no separate CORS-enabled deployment to keep in sync.
_FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sahm_finance_platform.html"
)


@app.get("/", include_in_schema=False)
def serve_frontend():
    from fastapi.responses import FileResponse

    return FileResponse(_FRONTEND_PATH, media_type="text/html")


# ---- WATCHLIST STORAGE (SQLite) ---------------------------------------
# The watchlist is per-user (keyed by the signed-in Google account's email)
# and needs to sync across devices, so it lives in a small local SQLite
# database on this server instead of the browser's localStorage. This is
# completely separate from Databricks/MOCK_MODE — it works either way.
WATCHLIST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saia_app.db")


def get_db_connection():
    conn = sqlite3.connect(WATCHLIST_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_watchlist_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                email TEXT NOT NULL,
                ticker TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (email, ticker)
            )
            """
        )
        # Saved news works exactly like the stock watchlist above, but keyed
        # by article_id instead of ticker — one row per (user, saved article).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_news (
                email TEXT NOT NULL,
                article_id TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (email, article_id)
            )
            """
        )
        # Persistent chat log per signed-in email — one row per message (both
        # the user's turn and the assistant's reply), so a user who signs
        # back in (even on another device) sees their past conversation with
        # SAIA instead of starting fresh. This is separate from chatbot.py's
        # MemorySaver, which only keeps history in memory for the LangGraph
        # agent's own conversational context and is lost on restart — this
        # table is what the frontend actually reads to redraw the chat log UI.
        # A user can now have several separate conversations with SAIA (like
        # ChatGPT's sidebar) instead of one single never-ending log. Each row
        # here is one conversation; chat_messages below points at one of
        # these via conversation_id instead of just being grouped by email.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_conversations (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_conversations_email ON chat_conversations (email, updated_at DESC)"
        )
        # Persistent chat log per signed-in email — one row per message (both
        # the user's turn and the assistant's reply), so a user who signs
        # back in (even on another device) sees their past conversation with
        # SAIA instead of starting fresh. This is separate from chatbot.py's
        # MemorySaver, which only keeps history in memory for the LangGraph
        # agent's own conversational context and is lost on restart — this
        # table is what the frontend actually reads to redraw the chat log UI.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        # conversation_id didn't exist in the very first version of this
        # table — add it on top of any DB that was already created, rather
        # than requiring a fresh database.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
        if "conversation_id" not in existing_cols:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN conversation_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_email ON chat_messages (email, id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages (conversation_id, id)"
        )
        # Durable per-user facts (name, preferences, etc.) extracted from chat
        # by chatbot.py — one row per signed-in email, shared across every
        # conversation that email has (unlike chat_messages/chat_conversations,
        # which are scoped to one conversation_id at a time). This is what
        # lets SAIA "remember" things like the user's name even in a brand
        # new conversation started from the sidebar.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_facts (
                email TEXT PRIMARY KEY,
                facts TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
        _backfill_legacy_chat_conversations(conn)


def _backfill_legacy_chat_conversations(conn):
    """Messages saved before conversations existed have conversation_id =
    NULL. Group each such email's orphaned messages into one "legacy"
    conversation instead of just losing/hiding them, so nothing a user
    already said to SAIA disappears from their sidebar."""
    emails = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT email FROM chat_messages WHERE conversation_id IS NULL"
        ).fetchall()
    ]
    for email in emails:
        first_user_msg = conn.execute(
            "SELECT content FROM chat_messages WHERE email = ? AND conversation_id IS NULL "
            "AND role = 'user' ORDER BY id LIMIT 1",
            (email,),
        ).fetchone()
        title = _make_conversation_title(first_user_msg[0]) if first_user_msg else "Previous conversation"
        conv_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO chat_conversations (id, email, title) VALUES (?, ?, ?)",
            (conv_id, email, title),
        )
        conn.execute(
            "UPDATE chat_messages SET conversation_id = ? WHERE email = ? AND conversation_id IS NULL",
            (conv_id, email),
        )
    if emails:
        conn.commit()


def _make_conversation_title(message: str) -> str:
    """Auto-titles a new conversation from its first user message, the same
    way ChatGPT/Claude do — trimmed to a short one-liner for the sidebar."""
    flat = " ".join((message or "").strip().split())
    if not flat:
        return "New conversation"
    return (flat[:42] + "…") if len(flat) > 42 else flat


init_watchlist_db()


class WatchlistToggleRequest(BaseModel):
    email: str
    ticker: str


class SavedNewsToggleRequest(BaseModel):
    email: str
    article_id: str


class NewConversationRequest(BaseModel):
    email: str


class ChatRequest(BaseModel):
    message: str
    # Groups messages into one ongoing conversation. Pass the signed-in
    # user's email when you have it; falls back to email, then "default",
    # so anonymous visitors still get a working (if shared) chat.
    thread_id: Optional[str] = None
    email: Optional[str] = None
    # Which sidebar conversation this message belongs to. Signed-in users
    # should pass the id of the conversation currently open; if omitted
    # (e.g. their very first-ever message), the backend creates one for them
    # automatically and returns its id in the response.
    conversation_id: Optional[str] = None

# Allow the frontend to call this API. Tighten allow_origins to your actual
# frontend origin(s) once you're not just testing locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ---- CONNECTION POOL ----------------------------------------------------
# We used to keep ONE shared connection behind a global lock. That made any
# two requests that happened to land at the same time (e.g. bootstrap firing
# off /api/companies + /api/news + the home news preview together) queue up
# and run one-at-a-time against Databricks, even though they touch different
# tables and have nothing to do with each other. Total wait time became the
# SUM of every query instead of the time of the slowest one.
#
# A small pool of warm connections fixes that: each request grabs whichever
# connection is free, so independent queries actually run in parallel. The
# pool is still bounded (POOL_SIZE) so we don't open unlimited connections
# to the warehouse under heavy concurrent load.
POOL_SIZE = int(os.environ.get("DATABRICKS_POOL_SIZE", "4"))
_connection_pool: "queue.Queue" = queue.Queue()
_pool_lock = threading.Lock()
_pool_initialized = False


def _open_connection():
    return sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def _init_pool():
    global _pool_initialized
    with _pool_lock:
        if _pool_initialized:
            return
        # Slots start as None (lazily opened on first use) so app startup
        # doesn't have to pay for POOL_SIZE handshakes up front.
        for _ in range(POOL_SIZE):
            _connection_pool.put(None)
        _pool_initialized = True


def run_query(query: str, params: Optional[dict] = None):
    """Run one query against Databricks using a connection from the pool.

    Opening a brand-new Databricks SQL connection per request is expensive
    (each handshake can take seconds, especially against a warehouse that
    isn't already warm) — so connections are opened once and reused. Unlike
    before, up to POOL_SIZE requests can now run at the same time on their
    own connection instead of all serializing behind one lock.
    """
    _init_pool()
    conn = _connection_pool.get()  # waits here if all POOL_SIZE conns are busy
    try:
        for attempt in (1, 2):
            try:
                if conn is None:
                    conn = _open_connection()
                with conn.cursor() as cursor:
                    cursor.execute(query, params or {})
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
            except Exception:
                # Connection likely dropped/expired — drop it and retry once
                # with a fresh one before giving up.
                conn = None
                if attempt == 2:
                    raise
    finally:
        # Always return a slot to the pool (even a None one, which will be
        # lazily reopened next time it's drawn) so the pool never leaks down.
        _connection_pool.put(conn)



# ---- SIMPLE IN-MEMORY TTL CACHE ---------------------------------------
# The frontend polls /api/companies every 20s (per open browser tab, see
# LIVE_REFRESH_INTERVAL_MS in sahm_finance_platform.html) to flash live
# price moves on the ticker tape. Prices in gold.fact_stock_prices only
# actually change once a day (end-of-day batch load), so re-running the
# same expensive multi-CTE query against the Databricks warehouse on every
# single poll — from every open tab, from every user — burns SQL warehouse
# time for data that hasn't moved. This cache makes the warehouse do the
# work at most once every COMPANIES_CACHE_TTL_SECONDS, no matter how many
# tabs/users are polling in that window.
_companies_cache = {"data": None, "expires_at": 0.0}
COMPANIES_CACHE_TTL_SECONDS = 20  # matches the frontend's poll interval


@app.get("/api/companies")
def get_companies():
    """Maps to gold.dim_companies — one snapshot row per ticker.

    dim_companies stopped carrying live price/change/volume columns and
    picked up a bunch of new descriptive ones instead (short_name, exchange,
    currency, country, sector, industry, quote_type, shares_outstanding,
    market_cap_usd, market_cap_reference_raw, gold_loaded_at...). So the
    live numbers (price, change, change_percent, volume, avg_volume_3m) are
    now derived here from gold.fact_stock_prices directly:
      - price / volume          -> the most recent trade_date per ticker
      - change / change_percent -> most recent close vs. the one before it
      - avg_volume_3m           -> average daily volume over the last ~90 days
    """
    if MOCK_MODE:
        # IMPORTANT: don't just return companies.json's own "price" /
        # "change" / "change_percent" fields as-is. Those are a separate,
        # independently-generated static snapshot from stock_prices.json
        # (the file that actually feeds the price chart / get_stock_prices),
        # so the two can silently drift apart — e.g. the chatbot/company
        # card shows one price while the chart's latest point shows another
        # for the same ticker. The Databricks branch below never has this
        # problem because both numbers come from the same
        # fact_stock_prices table. To match that behavior here, we derive
        # price/change/change_percent/volume from stock_prices.json's own
        # last two data points instead, so both are always consistent by
        # construction — companies.json only supplies the non-price
        # descriptive fields (name, sector, market cap, etc.).
        companies = _load_json("companies.json")
        history = _load_json("stock_prices.json")
        AVG_VOLUME_WINDOW_DAYS = 90
        out = []
        for c in companies:
            row = dict(c)
            prices = history.get(row.get("ticker"), []) or []
            if prices:
                latest = prices[-1]
                prev = prices[-2] if len(prices) >= 2 else None
                price = latest.get("close_price")
                prev_close = prev.get("close_price") if prev else None
                row["price"] = price
                row["volume"] = latest.get("volume")
                if price is not None and prev_close:
                    row["change"] = price - prev_close
                    row["change_percent"] = (price - prev_close) / prev_close * 100
                else:
                    row["change"] = None
                    row["change_percent"] = None
                recent = prices[-AVG_VOLUME_WINDOW_DAYS:]
                volumes = [p.get("volume") for p in recent if p.get("volume") is not None]
                row["avg_volume_3m"] = sum(volumes) / len(volumes) if volumes else None
            # else: no price history for this ticker in mock data — leave
            # whatever companies.json had (or missing/None) rather than
            # guessing.
            out.append(row)
        return out

    import time

    now = time.time()
    if _companies_cache["data"] is not None and now < _companies_cache["expires_at"]:
        return _companies_cache["data"]

    query = f"""
        WITH ranked_prices AS (
            SELECT
                ticker, trade_date, close_price, volume,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
            FROM {CATALOG}.{SCHEMA}.fact_stock_prices
        ),
        price_stats AS (
            SELECT
                ticker,
                MAX(CASE WHEN rn = 1 THEN close_price END) AS price,
                MAX(CASE WHEN rn = 1 THEN volume END)      AS volume,
                MAX(CASE WHEN rn = 2 THEN close_price END) AS prev_close
            FROM ranked_prices
            WHERE rn <= 2
            GROUP BY ticker
        ),
        avg_volume AS (
            SELECT ticker, AVG(volume) AS avg_volume_3m
            FROM {CATALOG}.{SCHEMA}.fact_stock_prices
            WHERE trade_date >= date_sub(current_date(), 90)
            GROUP BY ticker
        )
        SELECT
            d.ticker, d.company_name, d.short_name, d.exchange, d.currency,
            d.country, d.sector, d.industry, d.quote_type, d.shares_outstanding,
            d.category, d.market_cap_usd AS market_cap, d.pe_ratio,
            d.week52_change_percent, d.week52_low, d.week52_high,
            ps.price,
            (ps.price - ps.prev_close) AS change,
            CASE
                WHEN ps.prev_close IS NULL OR ps.prev_close = 0 THEN NULL
                ELSE (ps.price - ps.prev_close) / ps.prev_close * 100
            END AS change_percent,
            ps.volume,
            av.avg_volume_3m
        FROM {CATALOG}.{SCHEMA}.dim_companies d
        LEFT JOIN price_stats ps ON ps.ticker = d.ticker
        LEFT JOIN avg_volume  av ON av.ticker = d.ticker
    """
    try:
        data = run_query(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load companies: {e}")
    _companies_cache["data"] = data
    _companies_cache["expires_at"] = time.time() + COMPANIES_CACHE_TTL_SECONDS
    return data


@app.get("/api/stock-prices")
def get_stock_prices(
    ticker: str = Query(..., description="Stock ticker, e.g. AAPL"),
    days: int = Query(60, ge=1, le=3650, description="How many most-recent trading days to return"),
):
    """Maps to gold.fact_stock_prices — most recent `days` of daily history for one ticker, oldest first.

    We used to pull the FULL history (years of rows) on every request, which
    is why the page felt slow — the table and the prediction model only ever
    look at the last ~30-60 days anyway.
    """
    if MOCK_MODE:
        all_history = _load_json("stock_prices.json")
        return (all_history.get(ticker, []) or [])[-days:]
    query = f"""
        SELECT trade_date, open_price, high_price, low_price,
               close_price, adjusted_close_price, volume
        FROM {CATALOG}.{SCHEMA}.fact_stock_prices
        WHERE ticker = %(ticker)s
        ORDER BY trade_date DESC
        LIMIT {days}
    """
    try:
        rows = run_query(query, {"ticker": ticker})
        rows.reverse()  # query above is newest-first (needed for LIMIT); flip back to oldest-first
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stock prices: {e}")
@app.get("/api/config")
def get_config():
    return {
        "google_client_id": os.environ["GOOGLE_CLIENT_ID"]
    }
@app.get("/api/stock-prices/download")
def download_stock_prices(
    days: int = Query(30, ge=1, le=3650, description="How many most-recent trading days per ticker to include"),
):
    """Bulk export for the 'Download Stock Data (CSV)' button.

    The frontend used to build this file by calling /api/stock-prices once
    PER TICKER (14k+ tickers => 14k+ sequential round trips, all serialized
    through the single shared Databricks connection in run_query) — which is
    why the download would sit for a long time and then fail outright rather
    than actually finish. One query with a per-ticker window function returns
    everything at once, the same way /api/news/download already does for news.
    """
    if MOCK_MODE:
        history = _load_json("stock_prices.json")
        companies = {c["ticker"]: c for c in _load_json("companies.json")}
        rows = []
        for ticker, prices in history.items():
            name = companies.get(ticker, {}).get("company_name", "")
            for p in (prices or [])[-days:]:
                rows.append({"ticker": ticker, "company_name": name, **p})
        return rows
    query = f"""
        WITH ranked AS (
            SELECT
                fp.ticker, d.company_name, fp.trade_date, fp.open_price, fp.high_price,
                fp.low_price, fp.close_price, fp.adjusted_close_price, fp.volume,
                ROW_NUMBER() OVER (PARTITION BY fp.ticker ORDER BY fp.trade_date DESC) AS rn
            FROM {CATALOG}.{SCHEMA}.fact_stock_prices fp
            JOIN {CATALOG}.{SCHEMA}.dim_companies d ON d.ticker = fp.ticker
        )
        SELECT ticker, company_name, trade_date, open_price, high_price,
               low_price, close_price, adjusted_close_price, volume
        FROM ranked
        WHERE rn <= {days}
        ORDER BY ticker, trade_date
    """
    try:
        return run_query(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stock prices for download: {e}")


@app.get("/api/news/download")
def download_news(
    days: int = Query(3650, ge=1, le=3650, description="Only include articles published within the last N days (default covers all available history)"),
):
    """Maps to gold.news_sentiments — same article fields as before, plus the
    per-article FinBERT sentiment scores (positive_score / negative_score /
    neutral_score) now that fact_news was replaced by this table.

    `days` lets the "Download News (CSV)" button's duration picker (24h / 7d /
    30d / 90d / all) filter by each article's published_at, the same way
    /api/stock-prices/download already filters price history by days."""
    if MOCK_MODE:
        from datetime import datetime, timedelta, timezone
        all_news = _load_json("news.json")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        def _parse(ts):
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                # published_at in the test-data JSON isn't always timezone-aware
                # (e.g. "2026-07-01T10:30:00" with no offset). Comparing that
                # directly against an aware `cutoff` raises TypeError, which
                # crashes this endpoint with an unhandled 500. Treat naive
                # timestamps as UTC so the comparison always succeeds.
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None
        try:
            return [n for n in all_news if (_parse(n.get("published_at")) or cutoff) >= cutoff]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download news: {e}")
    query = f"""
        SELECT
            article_id,
            ticker,
            related_tickers,
            related_tickers_count,
            title,
            description,
            publisher_name,
            source_url,
            published_at,
            positive_score,
            negative_score,
            neutral_score
        FROM {CATALOG}.{SCHEMA}.news_sentiments
        WHERE published_at >= current_timestamp() - INTERVAL {days} DAYS
    """

    try:
        rows = run_query(query)

        for row in rows:
            # related_tickers comes back as a Databricks/Arrow array type, not
            # a plain list. `if rt` calls bool() on it, and numpy/Arrow arrays
            # with more than one element raise "The truth value of an array
            # with more than one element is ambiguous" instead of returning
            # True/False. Use `is not None` instead, same as /api/news and
            # /api/news/by-ids already do.
            rt = row.get("related_tickers")
            row["related_tickers"] = list(rt) if rt is not None else []

            pub = row.get("published_at")
            row["published_at"] = (
                pub.isoformat() if hasattr(pub, "isoformat") else pub
            )

        return rows

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download news: {e}"
        )
@app.get("/api/news/by-ids")
def get_news_by_ids(
    article_ids: str = Query(..., description="Comma-separated article_ids to fetch, e.g. from a user's saved-news list"),
):
    """Maps to gold.news_sentiments — fetches the exact article rows for a
    given set of article_ids, regardless of publish date or where they fall
    in the paginated /api/news feed.

    This exists for the Saved News view: SAVED_NEWS on the frontend can
    contain article_ids the user saved a while ago, which may no longer be
    among the most-recent `limit`/`offset` window /api/news returns. Filtering
    against whatever page of /api/news happens to be loaded made saved
    articles silently disappear once they aged out of that window — this
    endpoint looks them up directly instead.
    """
    ids = [i for i in article_ids.split(",") if i]
    if not ids:
        return []
    if MOCK_MODE:
        all_news = _load_json("news.json")
        return [n for n in all_news if n["article_id"] in ids]
    placeholders = ", ".join(f"%(id_{i})s" for i in range(len(ids)))
    params = {f"id_{i}": v for i, v in enumerate(ids)}
    query = f"""
        SELECT article_id, ticker, related_tickers, related_tickers_count,
               title, description, publisher_name, source_url, published_at,
               positive_score, negative_score, neutral_score
        FROM {CATALOG}.{SCHEMA}.news_sentiments
        WHERE article_id IN ({placeholders})
    """
    try:
        rows = run_query(query, params)
        for row in rows:
            rt = row.get("related_tickers")
            row["related_tickers"] = list(rt) if rt is not None else []
            pub = row.get("published_at")
            row["published_at"] = pub.isoformat() if hasattr(pub, "isoformat") else pub
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load news by ids: {e}")


@app.get("/api/news")
def get_news(
    ticker: Optional[str] = Query(None, description="Filter by ticker; omit for all news"),
    limit: int = Query(200, ge=1, le=1000, description="Max articles to return"),
    offset: int = Query(0, ge=0, description="How many articles to skip (for 'load more' pagination)"),
):
    """Maps to gold.news_sentiments — title + description + source_url per
    article, plus the per-article FinBERT sentiment scores (positive_score /
    negative_score / neutral_score). This replaced gold.fact_news, which
    didn't carry sentiment at all.

    news_sentiments is written newest-first by the ETL, but that's a
    physical/storage-order assumption, not a guarantee — Delta/Spark reads
    can reorder rows across files (parallel reads, OPTIMIZE/Z-ORDER/VACUUM
    compaction, etc.), and pagination (LIMIT/OFFSET) silently breaks if the
    order shifts between requests. Explicit ORDER BY published_at DESC costs
    a small amount of query time but keeps the feed (and the stock detail
    page's "Related News" section, which also calls this via `ticker`)
    correctly ordered regardless of storage layout."""
    if MOCK_MODE:
        all_news = _load_json("news.json")
        if ticker:
            all_news = [
                n for n in all_news
                if n["ticker"] == ticker or ticker in (n.get("related_tickers") or [])
            ]
        return all_news[offset:offset + limit]
    base_query = f"""
        SELECT article_id, ticker, related_tickers, related_tickers_count,
               title, description, publisher_name, source_url, published_at,
               positive_score, negative_score, neutral_score
        FROM {CATALOG}.{SCHEMA}.news_sentiments
    """
    if ticker:
        query = base_query + f" WHERE ticker = %(ticker)s ORDER BY published_at DESC LIMIT {limit} OFFSET {offset}"
        params = {"ticker": ticker}
    else:
        query = base_query + f" ORDER BY published_at DESC LIMIT {limit} OFFSET {offset}"
        params = {}
    try:
        rows = run_query(query, params)
        for row in rows:
            # related_tickers comes back as a Databricks/Arrow array type —
            # convert to a plain Python list so FastAPI can serialize it.
            rt = row.get("related_tickers")
            row["related_tickers"] = list(rt) if rt is not None else []
            # published_at comes back as a datetime/Timestamp object —
            # convert to an ISO string.
            pub = row.get("published_at")
            row["published_at"] = pub.isoformat() if hasattr(pub, "isoformat") else pub
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load news: {e}")


@app.get("/api/news/search")
def search_news(
    q: str = Query(..., min_length=1, description="Free-text search over title, ticker, and publisher name"),
    limit: int = Query(20, ge=1, le=200, description="Max articles to return per page"),
    offset: int = Query(0, ge=0, description="How many matching articles to skip (for 'load more' pagination)"),
):
    """Maps to gold.news_sentiments — same shape as /api/news, but filtered
    server-side by `q` instead of client-side.

    The News page's free-text search used to work by pulling a "search pool"
    of articles into the browser (starting at 2000, then paging in another
    1000 at a time via /api/news's LIMIT/OFFSET whenever a rare query didn't
    have enough matches yet) and filtering them in memory. A rare search term
    could end up walking deep into a table with far more rows than
    stock_price_predictions ever had, repeatedly holding one of the
    POOL_SIZE=4 Databricks connections busy long enough to slow down (or
    time out) unrelated requests across the site — news is too large to just
    fetch in one unbounded query the way predictions did, so instead of
    avoiding OFFSET entirely, this pushes the filtering into SQL.

    WHERE narrows the table down to matching rows FIRST, so LIMIT/OFFSET only
    ever pages through that much smaller matching set, not the entire table,
    no matter how deep 'Load more' pages in. ORDER BY published_at DESC is
    explicit here too — see /api/news's docstring for why relying on
    storage order alone isn't safe under Delta/Spark."""
    if MOCK_MODE:
        all_news = _load_json("news.json")
        ql = q.lower()
        matches = [
            n for n in all_news
            if ql in (n.get("title") or "").lower()
            or ql in (n.get("ticker") or "").lower()
            or ql in (n.get("publisher_name") or "").lower()
        ]
        return matches[offset:offset + limit]
    like = f"%{q.lower()}%"
    query = f"""
        SELECT article_id, ticker, related_tickers, related_tickers_count,
               title, description, publisher_name, source_url, published_at,
               positive_score, negative_score, neutral_score
        FROM {CATALOG}.{SCHEMA}.news_sentiments
        WHERE LOWER(title) LIKE %(like)s
           OR LOWER(ticker) LIKE %(like)s
           OR LOWER(publisher_name) LIKE %(like)s
        ORDER BY published_at DESC
        LIMIT {limit}
        OFFSET {offset}
    """
    try:
        rows = run_query(query, {"like": like})
        for row in rows:
            rt = row.get("related_tickers")
            row["related_tickers"] = list(rt) if rt is not None else []
            pub = row.get("published_at")
            row["published_at"] = pub.isoformat() if hasattr(pub, "isoformat") else pub
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search news: {e}")


@app.get("/api/predictions/all")
def get_all_predictions():
    """Maps to gold.stock_price_predictions — the ENTIRE table in one query,
    no LIMIT/OFFSET.

    Exists for the frontend's predictions search pool. That used to page
    through /api/predictions with LIMIT 1000 OFFSET N in a loop, but OFFSET
    pagination gets more expensive with every page — each call re-sorts the
    WHOLE table from scratch, then discards the first `offset` rows just to
    reach the next chunk, so pulling all ~13k rows that way costs roughly
    O(n^2) total DB work across all the calls combined. One unbounded query
    like this (matching how /api/companies already fetches everything in a
    single round trip) sorts once and returns everything: O(n log n), a
    single request, done. No ORDER BY here at all, since a search pool only
    needs to be filtered client-side, not ranked — skipping the sort saves
    even that O(n log n) cost server-side.

    Not used for the normal paginated browse view (that stays on
    /api/predictions with LIMIT/OFFSET, since browsing only ever needs the
    top-ranked page or two before the user stops scrolling)."""
    if MOCK_MODE:
        return _load_json("predictions.json")
    query = f"""
        SELECT
            ticker, company_name,
            last_known_close_price, predicted_close_price,
            CASE
                WHEN last_known_close_price IS NULL OR last_known_close_price = 0
                     OR predicted_close_price IS NULL THEN NULL
                ELSE (predicted_close_price - last_known_close_price) / last_known_close_price * 100
            END AS predicted_change_percent,
            train_rows_used, pipeline_name, pipeline_version, prediction_generated_at
        FROM {CATALOG}.{SCHEMA}.stock_price_predictions
    """
    try:
        rows = run_query(query)
        for row in rows:
            gen = row.get("prediction_generated_at")
            row["prediction_generated_at"] = gen.isoformat() if hasattr(gen, "isoformat") else gen
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load all predictions: {e}")


@app.get("/api/predictions")
def get_predictions(
    limit: int = Query(200, ge=1, le=1000, description="Max tickers to return per page, sorted by strongest predicted gain first"),
    offset: int = Query(0, ge=0, description="How many tickers to skip (for 'load more' pagination past the first `limit` rows)"),
    ticker: Optional[str] = Query(None, description="Return the prediction for this exact ticker only, regardless of its rank"),
):
    """Maps to gold.stock_price_predictions — one row per ticker holding the
    latest ML-generated 30-day price prediction.

    predicted_change_percent is derived here (not stored) from
    predicted_close_price vs. last_known_close_price, the same price the
    prediction was actually computed against, so the percentage always lines
    up with the two prices shown next to it.

    NOTE: this deliberately does NOT LEFT JOIN dim_companies to backfill
    tickers that have no prediction yet (recent IPOs/spinoffs without enough
    fact_stock_prices history — confirmed gap: 13,599 companies vs 13,340
    predictions). An earlier version of this endpoint did that join so those
    ~259 companies wouldn't silently disappear from the API, but joining +
    sorting 13k+ rows on every request — especially the 14 sequential calls
    the frontend's "search all predictions" pool makes — held DATABRICKS_TOKEN
    connections (POOL_SIZE=4, shared with every other endpoint) for long
    enough to starve /api/news and search across the whole site. The
    dim_companies join for that gap is now done client-side instead (frontend
    already has the full company list loaded for free) — see
    joinPredictionRows/missing-company merge in the frontend — so this query
    stays cheap and fast.

    limit is capped at 1000 per request — if the platform has more tickers
    than that, callers MUST page through with `offset` (same pattern as
    /api/news) to reach the rest, or pass `ticker` to look up one specific
    company directly instead of hunting for it inside a capped page.
    """
    if MOCK_MODE:
        rows = _load_json("predictions.json")
        if ticker:
            return [r for r in rows if r.get("ticker") == ticker]
        return rows[offset:offset + limit]
    where_clause = ""
    params = {}
    if ticker:
        where_clause = "WHERE ticker = %(ticker)s"
        params["ticker"] = ticker
    query = f"""
        SELECT
            ticker, company_name,
            last_known_close_price, predicted_close_price,
            CASE
                WHEN last_known_close_price IS NULL OR last_known_close_price = 0
                     OR predicted_close_price IS NULL THEN NULL
                ELSE (predicted_close_price - last_known_close_price) / last_known_close_price * 100
            END AS predicted_change_percent,
            train_rows_used, pipeline_name, pipeline_version, prediction_generated_at
        FROM {CATALOG}.{SCHEMA}.stock_price_predictions
        {where_clause}
        ORDER BY predicted_change_percent DESC NULLS LAST
        LIMIT {limit}
        OFFSET {offset}
    """
    try:
        rows = run_query(query, params)
        for row in rows:
            gen = row.get("prediction_generated_at")
            row["prediction_generated_at"] = gen.isoformat() if hasattr(gen, "isoformat") else gen
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load predictions: {e}")


@app.get("/api/watchlist")
def get_watchlist(email: str = Query(..., description="Signed-in user's Google account email")):
    """Returns the list of tickers this email has saved to their watchlist."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT ticker FROM watchlist WHERE email = ? ORDER BY added_at",
            (email,),
        ).fetchall()
    return {"watchlist": [row[0] for row in rows]}


@app.post("/api/watchlist/toggle")
def toggle_watchlist(body: WatchlistToggleRequest):
    """Adds the ticker if it's not already saved for this email, removes it if it is.
    Returns the updated full watchlist for that email."""
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE email = ? AND ticker = ?",
            (body.email, body.ticker),
        ).fetchone()
        if existing:
            conn.execute(
                "DELETE FROM watchlist WHERE email = ? AND ticker = ?",
                (body.email, body.ticker),
            )
        else:
            conn.execute(
                "INSERT INTO watchlist (email, ticker) VALUES (?, ?)",
                (body.email, body.ticker),
            )
        conn.commit()
        rows = conn.execute(
            "SELECT ticker FROM watchlist WHERE email = ? ORDER BY added_at",
            (body.email,),
        ).fetchall()
    return {"watchlist": [row[0] for row in rows]}


@app.get("/api/saved-news")
def get_saved_news(email: str = Query(..., description="Signed-in user's Google account email")):
    """Returns the list of article_ids this email has saved — same idea as
    /api/watchlist, just for news articles instead of tickers."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT article_id FROM saved_news WHERE email = ? ORDER BY added_at",
            (email,),
        ).fetchall()
    return {"saved_news": [row[0] for row in rows]}


@app.post("/api/saved-news/toggle")
def toggle_saved_news(body: SavedNewsToggleRequest):
    """Adds the article if it's not already saved for this email, removes it
    if it is. Returns the updated full list of saved article_ids for that
    email — mirrors /api/watchlist/toggle."""
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM saved_news WHERE email = ? AND article_id = ?",
            (body.email, body.article_id),
        ).fetchone()
        if existing:
            conn.execute(
                "DELETE FROM saved_news WHERE email = ? AND article_id = ?",
                (body.email, body.article_id),
            )
        else:
            conn.execute(
                "INSERT INTO saved_news (email, article_id) VALUES (?, ?)",
                (body.email, body.article_id),
            )
        conn.commit()
        rows = conn.execute(
            "SELECT article_id FROM saved_news WHERE email = ? ORDER BY added_at",
            (body.email,),
        ).fetchall()
    return {"saved_news": [row[0] for row in rows]}


def get_user_facts(email: Optional[str]) -> list:
    """Durable facts SAIA has learned about this email across ALL of their
    conversations (name, stated preferences, etc.) — see chatbot.py's fact
    extraction. Returns [] for anonymous users or a user we know nothing
    about yet."""
    if not email:
        return []
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT facts FROM user_facts WHERE email = ?", (email,)
            ).fetchone()
        return json.loads(row[0]) if row else []
    except Exception as e:
        print(f"[user-facts] failed to load facts for {email}: {e}")
        return []


def save_user_facts(email: str, facts: list) -> None:
    """Best-effort overwrite of the stored fact list for this email. Never
    raises — a failed save just means we didn't learn something new this
    turn, which is far less bad than crashing the chat response."""
    if not email:
        return
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_facts (email, facts, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(email) DO UPDATE SET
                    facts = excluded.facts,
                    updated_at = excluded.updated_at
                """,
                (email, json.dumps(facts, ensure_ascii=False)),
            )
            conn.commit()
    except Exception as e:
        print(f"[user-facts] failed to save facts for {email}: {e}")


def _get_recent_chat_history(conversation_id: Optional[str], limit: int = 200) -> list:
    """Shared by /api/chat-history (frontend redraw) and /api/chat (feeding
    the LLM real context on a cold thread — see chatbot.py's
    _seed_history_if_cold). Returns oldest-first for one conversation."""
    if not conversation_id:
        return []
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at FROM (
                SELECT role, content, created_at, id
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (conversation_id, limit),
        ).fetchall()
    return [{"role": row[0], "content": row[1], "created_at": row[2]} for row in rows]


@app.get("/api/chat-conversations")
def list_chat_conversations(
    email: str = Query(..., description="Signed-in user's Google account email"),
):
    """Returns this email's conversations for the sidebar, most-recently
    active first (a conversation's updated_at bumps every time a message is
    saved to it — see _save_chat_message)."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chat_conversations "
            "WHERE email = ? ORDER BY updated_at DESC",
            (email,),
        ).fetchall()
    return {
        "conversations": [
            {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows
        ]
    }


@app.post("/api/chat-conversations")
def create_chat_conversation(body: NewConversationRequest):
    """Explicit "start a new conversation" action (the sidebar's + button).
    Also used implicitly by /api/chat itself when a signed-in user's very
    first message doesn't carry a conversation_id yet."""
    conv_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_conversations (id, email, title) VALUES (?, ?, 'New conversation')",
            (conv_id, body.email),
        )
        conn.commit()
    return {"id": conv_id, "title": "New conversation"}


@app.delete("/api/chat-conversations/{conversation_id}")
def delete_chat_conversation(
    conversation_id: str,
    email: str = Query(..., description="Signed-in user's Google account email"),
):
    """Deletes one conversation and all of its messages — the sidebar's
    trash-icon action (after the frontend's own inline confirm step; this
    endpoint itself doesn't ask again). Checks the conversation actually
    belongs to this email first, so one signed-in user can't delete
    another's conversation just by guessing/replaying an id."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT email FROM chat_conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if row[0] != email:
            raise HTTPException(status_code=403, detail="This conversation doesn't belong to this account")
        conn.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM chat_conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    return {"ok": True}


@app.get("/api/chat-history")
def get_chat_history(
    email: str = Query(..., description="Signed-in user's Google account email"),
    conversation_id: Optional[str] = Query(
        None, description="Which conversation to fetch; defaults to this email's most recently active one"
    ),
    limit: int = Query(200, ge=1, le=2000, description="Max number of most-recent messages to return"),
):
    """Returns one conversation's message log, oldest first, so the frontend
    can redraw it when the user opens it (sign-in, refresh, or clicking it
    in the sidebar). Falls back to the most recently active conversation if
    conversation_id isn't given, so old callers/bookmarks still work."""
    with get_db_connection() as conn:
        if not conversation_id:
            row = conn.execute(
                "SELECT id FROM chat_conversations WHERE email = ? ORDER BY updated_at DESC LIMIT 1",
                (email,),
            ).fetchone()
            conversation_id = row[0] if row else None
    return {
        "history": _get_recent_chat_history(conversation_id, limit),
        "conversation_id": conversation_id,
    }


def _save_chat_message(email: str, conversation_id: Optional[str], role: str, content: str):
    """Best-effort append to the persistent chat log for this email/
    conversation. Never let a logging failure break the actual chat
    response — a lost history row is far less bad than the user's question
    failing outright."""
    if not email or not conversation_id:
        return
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO chat_messages (email, conversation_id, role, content) VALUES (?, ?, ?, ?)",
                (email, conversation_id, role, content),
            )
            # Keeps the sidebar sorted by recency (ORDER BY updated_at DESC
            # in list_chat_conversations) — the conversation you just used
            # floats back to the top, like every other chat app.
            conn.execute(
                "UPDATE chat_conversations SET updated_at = datetime('now') WHERE id = ?",
                (conversation_id,),
            )
            if role == "user":
                # Auto-title on the first user message, same as ChatGPT/Claude
                # ("New conversation" -> a short preview of what was asked).
                # Checked by count rather than by title text so a user who
                # happens to rename/retype something that starts with "New
                # conversation" doesn't get silently re-titled later.
                count = conn.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE conversation_id = ? AND role = 'user'",
                    (conversation_id,),
                ).fetchone()[0]
                if count == 1:
                    conn.execute(
                        "UPDATE chat_conversations SET title = ? WHERE id = ?",
                        (_make_conversation_title(content), conversation_id),
                    )
            conn.commit()
    except Exception as e:
        print(f"[chat-history] failed to save message for {email}/{conversation_id}: {e}")


def _update_user_facts_background(email: str, user_message: str, reply: str, existing_facts: list):
    """Runs on a background thread after a chat reply is already sent to the
    user. Asks chatbot.py's (small, tool-free) extraction call whether this
    turn contains a new durable fact about the user (name, stated
    preference, etc.), merges it into whatever we already knew, and saves
    it. Swallows all errors — this is a nice-to-have, never something that
    should surface as a user-facing failure."""
    try:
        updated = chatbot.extract_user_facts(existing_facts, user_message, reply)
        if updated != existing_facts:
            save_user_facts(email, updated)
    except Exception as e:
        print(f"[user-facts] background extraction failed for {email}: {e}")


@app.post("/api/chat")
def chat(body: ChatRequest):
    """Conversational endpoint for the SAIA assistant.

    Runs the message through the LangGraph + Gemini agent defined in
    chatbot.py, which can call back into this same file's data functions
    (companies, prices, news, predictions, watchlist) as tools instead of
    answering from the model's own memory.

    Each signed-in user can have several separate conversations (the
    sidebar). conversation_id picks which one this message belongs to; if
    it's missing (e.g. a brand new signed-in user's very first message) one
    is created automatically and returned in the response so the frontend
    can pick it up. thread_id scopes chatbot.py's own LangGraph memory and
    is derived from (email, conversation_id) so each conversation keeps its
    own independent context — anonymous chats (no email) fall back to an
    explicit thread_id from the client, then "default".

    Both the user's message and SAIA's reply are appended to the persistent
    chat_messages log for signed-in users (see /api/chat-history) —
    separate from chatbot.py's in-memory conversational context, this is
    what lets a user's chat history survive a server restart or a sign-in
    from another device. Anonymous chats (no email) are not logged.
    """
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    conversation_id = body.conversation_id
    if body.email and not conversation_id:
        # First-ever message from this signed-in user (or the frontend
        # otherwise didn't have one yet) — start a real conversation for it
        # instead of failing or silently dropping it.
        conversation_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO chat_conversations (id, email, title) VALUES (?, ?, 'New conversation')",
                (conversation_id, body.email),
            )
            conn.commit()

    thread_id = body.thread_id or (
        f"{body.email}:{conversation_id}" if body.email and conversation_id else "default"
    )
    # Read the persisted log BEFORE saving this turn's user message, so it
    # doesn't end up duplicated (once from history, once as the live
    # HumanMessage ask_chatbot adds for this turn). This is only ever
    # actually used by ask_chatbot the first time a thread_id is seen after
    # a process restart (see chatbot.py's _seed_history_if_cold) — on every
    # later turn the LangGraph checkpoint already has state and this is
    # ignored, so it's a cheap read, not a growing prompt.
    history = _get_recent_chat_history(conversation_id, limit=60) if conversation_id else None
    # Facts about this email learned in OTHER conversations (or earlier in
    # this one, from a previous process) — this is what lets a brand new
    # conversation from the sidebar still know the user's name etc., since
    # `history` above is scoped to just this one conversation_id.
    user_facts = get_user_facts(body.email)
    # Log the user's turn right away — even if the model call below fails,
    # what they asked is still worth keeping in their history.
    _save_chat_message(body.email, conversation_id, "user", body.message)
    try:
        reply = ask_chatbot(
            body.message, thread_id=thread_id, history=history, user_facts=user_facts
        )
        _save_chat_message(body.email, conversation_id, "bot", reply)
        if body.email:
            # Best-effort, non-blocking: look for new durable facts in this
            # turn and merge them in. Runs in the background so it never
            # slows down the reply the user is waiting for.
            threading.Thread(
                target=_update_user_facts_background,
                args=(body.email, body.message, reply, user_facts),
                daemon=True,
            ).start()
        return {"reply": reply, "thread_id": thread_id, "conversation_id": conversation_id}
    except Exception as e:
        msg = str(e)
        # chatbot.py now supports multiple Gemini keys (GEMINI_API_KEYS) with
        # automatic fallback — it only raises once no key is configured at
        # all, or every configured key has failed. Either case can still be
        # a quota/rate-limit issue (e.g. all keys hit their free-tier limit
        # at once), so we check the message content rather than the
        # exception type to decide which status/detail to send back.
        if isinstance(e, RuntimeError) and ("GOOGLE_API_KEY" in msg or "GEMINI_API_KEY" in msg) and "failed" not in msg.lower():
            # No key configured at all — a config problem, not a crash.
            raise HTTPException(status_code=503, detail=msg)
        # Gemini's free tier has tight per-minute/per-day quotas (see
        # https://ai.google.dev/gemini-api/docs/rate-limits). Surface a 429
        # with a clear, actionable message instead of a generic 500 — the
        # frontend already knows to show this text as-is in the chat log.
        # This also covers the "all configured keys failed" case when the
        # underlying reason across keys was quota/rate-limit related.
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    "The AI agent hit Gemini's free-tier rate limit (requests per "
                    "minute or per day) on every configured key. Wait a bit and try "
                    "again, add another API key, or raise the limit by enabling "
                    "billing in Google AI Studio."
                ),
            )
        if isinstance(e, RuntimeError):
            # e.g. "All N configured Gemini API key(s) failed" for
            # non-quota reasons (revoked/invalid keys, etc.)
            raise HTTPException(status_code=503, detail=msg)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE}


# Some deploy/monitoring setups probe "/v1/health" specifically (a common
# convention from versioned-API templates) rather than "/health" — alias it
# so those checks don't 404 even though this API isn't otherwise versioned.
@app.get("/v1/health")
def health_v1():
    return health()


# Chrome DevTools itself (not a real client of this API) automatically
# requests this well-known path on every page load to look for an optional
# project config file. It's harmless and unrelated to this app, but
# returning a quiet 204 here instead of a 404 keeps it out of the logs.
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    from fastapi import Response

    return Response(status_code=204)


# ---- CHATBOT -------------------------------------------------------------
# Imported at the bottom, after every function above it is defined, because
# chatbot.py imports this module back (`import app as backend`) to reuse
# get_companies/get_stock_prices/get_news/get_predictions/get_watchlist as
# tools instead of duplicating their MOCK_MODE / Databricks logic. Python
# only actually looks up `ask_chatbot` when /api/chat runs, by which point
# this module has finished loading either way.
import chatbot  # noqa: E402

ask_chatbot = chatbot.ask_chatbot
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("app:app", host="0.0.0.0", port=port)