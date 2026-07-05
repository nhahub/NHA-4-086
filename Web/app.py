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

Endpoints (used by the frontend's API_CONFIG):
  GET /api/companies
  GET /api/stock-prices?ticker=AAPL
  GET /api/news?ticker=AAPL      (ticker is optional — omit for all news)

Run it:
  1. pip install -r requirements.txt
  2. cp .env.example .env   and fill in your real Databricks values
  3. uvicorn app:app --reload --port 8000
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ---- MOCK MODE --------------------------------------------------------
# Set MOCK_MODE=true in .env (or the shell) to serve data straight from the
# JSON files in test_data/ instead of connecting to Databricks at all.
# Use this to confirm the frontend (sahm_finance_platform.html) reads the
# API correctly BEFORE you plug in real Databricks credentials.
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
    SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "silver")
else:
    from databricks import sql

    # ---- Databricks connection info (from .env — never hardcode these) ----
    DATABRICKS_SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]  # e.g. dbc-f88d29c6-4087.cloud.databricks.com
    DATABRICKS_HTTP_PATH = os.environ["DATABRICKS_HTTP_PATH"]              # e.g. /sql/1.0/warehouses/xxxxxxxxxxxxxxxx
    DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]                      # personal access token / service principal token
    CATALOG = os.environ.get("DATABRICKS_CATALOG", "finance_intelligence_hub")
    SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "silver")

app = FastAPI(title="SAIA Backend")

# Allow the frontend to call this API. Tighten allow_origins to your actual
# frontend origin(s) once you're not just testing locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def run_query(query: str, params: Optional[dict] = None):
    """Open a short-lived Databricks SQL connection, run one query, return rows as dicts."""
    with sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params or {})
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]


@app.get("/api/companies")
def get_companies():
    """Maps to silver.companies — one snapshot row per ticker."""
    if MOCK_MODE:
        return _load_json("companies.json")
    query = f"""
        SELECT ticker, company_name, category, price, change, change_percent,
               volume, avg_volume_3m, market_cap, pe_ratio,
               week52_change_percent, week52_low, week52_high
        FROM {CATALOG}.{SCHEMA}.companies
    """
    try:
        return run_query(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load companies: {e}")


@app.get("/api/stock-prices")
def get_stock_prices(ticker: str = Query(..., description="Stock ticker, e.g. AAPL")):
    """Maps to silver.stock_prices — full daily history for one ticker, oldest first."""
    if MOCK_MODE:
        all_history = _load_json("stock_prices.json")
        return all_history.get(ticker, [])
    query = f"""
        SELECT trade_date, open_price, high_price, low_price,
               close_price, adjusted_close_price, volume
        FROM {CATALOG}.{SCHEMA}.stock_prices
        WHERE ticker = %(ticker)s
        ORDER BY trade_date
    """
    try:
        return run_query(query, {"ticker": ticker})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stock prices: {e}")


@app.get("/api/news")
def get_news(ticker: Optional[str] = Query(None, description="Filter by ticker; omit for all news")):
    """Maps to silver.company_news_polygon — title + description + source_url per article."""
    if MOCK_MODE:
        all_news = _load_json("news.json")
        if ticker:
            return [
                n for n in all_news
                if n["ticker"] == ticker or ticker in (n.get("related_tickers") or [])
            ]
        return all_news
    base_query = f"""
        SELECT article_id, ticker, related_tickers, related_tickers_count,
               title, description, publisher_name, source_url, published_at
        FROM {CATALOG}.{SCHEMA}.company_news_polygon
    """
    if ticker:
        query = base_query + " WHERE ticker = %(ticker)s ORDER BY published_at DESC"
        params = {"ticker": ticker}
    else:
        query = base_query + " ORDER BY published_at DESC LIMIT 200"
        params = {}
    try:
        return run_query(query, params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load news: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE}
