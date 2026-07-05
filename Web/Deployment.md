# SAIA Backend

A small FastAPI proxy that sits between the SAIA frontend (`sahm_finance_platform.html`)
and your Databricks workspace. It exists because:

- A browser page can't call the Databricks REST API directly (no CORS support).
- A Databricks access token must never live in client-side (browser) code.

This server reads your Databricks connection info from `.env`, runs the SQL
against `finance_intelligence_hub.silver`, and returns plain JSON that the
frontend consumes.

## Setup

```bash
cd saia_backend
pip install -r requirements.txt
cp .env.example .env
# edit .env with your real DATABRICKS_SERVER_HOSTNAME / DATABRICKS_HTTP_PATH / DATABRICKS_TOKEN
uvicorn app:app --reload --port 8000
```

Check it's alive: open http://localhost:8000/health — should return `{"status":"ok"}`.

Then try:
- http://localhost:8000/api/companies
- http://localhost:8000/api/stock-prices?ticker=AAPL
- http://localhost:8000/api/news?ticker=AAPL

## Wiring up the frontend

In `sahm_finance_platform.html`, find `API_CONFIG` near the top of the `<script>`
block and set:

```js
const API_CONFIG = {
  baseUrl: 'http://localhost:8000', // or wherever you deploy this backend
  endpoints: { companies: '/api/companies', stockPrices: '/api/stock-prices', news: '/api/news' },
  useMockData: false // flip this once the backend above is running
};
```

That's the only change needed on the frontend side — every screen (Home,
Stocks, Predictions, News, and each company's detail page) already calls
`fetchCompanies()` / `fetchStockPrices(ticker)` / `fetchNews(ticker)`, which
route through this config.

## Tables this expects

- `silver.companies` — one row per ticker (price snapshot, market cap, P/E, 52-week range, etc.)
- `silver.stock_prices` — daily OHLC history per ticker
- `silver.company_news_polygon` — news articles with `title`, `description`, and `source_url`

If your column or table names differ, adjust the `SELECT` statements in `app.py`
accordingly — the frontend only cares about the JSON field names coming back,
which are documented in each endpoint's function in `app.py`.

## Deploying

Any host that can run a Python process works (a small VM, Render, Railway, an
Azure/AWS App Service, or — since you're already on Databricks — a
[Databricks App](https://docs.databricks.com/) can also host this and pull the
same env vars from Databricks' own secret management instead of a local `.env`).
