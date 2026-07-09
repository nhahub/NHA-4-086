"""
SAIA chatbot — a LangGraph agent, backed by Gemini, that can answer questions
about the data already served by this API (companies, prices, news,
predictions, watchlists) instead of guessing numbers from its own memory.

Design (the "chain"):

    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END
                 (no tool call) ---------------------------------> END

  * "agent"  : Gemini (via langchain-google-genai) with the tools below bound
               to it. It decides whether it needs data or can answer directly.
  * "tools"  : a prebuilt ToolNode that actually executes whichever tool(s)
               Gemini asked for and feeds the results back to the agent.
  * Memory   : a MemorySaver checkpointer keeps the full message history per
               `thread_id` (we use the user's email, or a random session id
               for anonymous visitors) so the chat is multi-turn without the
               frontend having to resend history on every request.
  * Trimming : before every call to Gemini we trim the stored history down to
               a token budget, so long conversations don't blow up context /
               cost — the checkpointer still keeps the untrimmed history.

This module is intentionally decoupled from FastAPI: `ask_chatbot()` is a
plain function app.py's /api/chat endpoint calls. That makes it trivial to
also expose over a CLI, a websocket, a queue worker, etc. later.
"""

import os
import sqlite3
import time
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Functions/config already defined in app.py — reused instead of duplicated,
# and instead of re-implementing the Databricks/MOCK_MODE branching here.
#
# This is imported LAZILY (inside _backend(), not at module top level) on
# purpose: app.py also imports this module (to expose ask_chatbot from
# /api/chat), so a top-level "import app" here would create a circular
# import that breaks depending on which of the two modules happens to be
# imported first. Deferring the import until a tool actually runs sidesteps
# that — by then both modules have finished loading no matter the order.
_backend_module = None


def _backend():
    global _backend_module
    if _backend_module is None:
        import app as backend

        _backend_module = backend
    return _backend_module

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ---- Multi-key support -----------------------------------------------------
# You can run this with more than one Gemini API key so that if one key hits
# its quota / rate limit / gets revoked, the chatbot automatically falls back
# to the next one instead of failing the request.
#
# Set GEMINI_API_KEYS to a comma-separated list, e.g.:
#   GEMINI_API_KEYS=AIzaSy-key-one,AIzaSy-key-two,AIzaSy-key-three
#
# The old single-key vars (GOOGLE_API_KEY / GEMINI_API_KEY) still work and
# are just treated as a one-key list, so nothing breaks for existing setups.
def _load_api_keys() -> list:
    multi = os.environ.get("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    if not keys:
        single = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if single:
            keys = [single.strip()]
    # de-dupe while preserving order, in case the same key got listed twice
    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


GEMINI_API_KEYS = _load_api_keys()
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None  # kept for backward compat


def _mask(key: str) -> str:
    """Never log a full key — just enough to tell keys apart in logs."""
    return f"...{key[-4:]}" if len(key) > 4 else "****"


# Errors from a key that's genuinely dead (quota exhausted, revoked,
# suspended, bad key) — these mean "move on to the next key". Errors that
# aren't about the key itself (a malformed request, a bug in our tool code)
# should NOT trigger key-hopping, since switching keys won't fix those and
# we'd just silently mask a real bug.
_KEY_FAILURE_SIGNALS = (
    "429",  # rate limit / quota
    "resource_exhausted",
    "quota",
    "403",  # permission denied
    "permission_denied",
    "401",  # unauthenticated
    "unauthenticated",
    "api key not valid",
    "api_key_invalid",
    "suspended",
)


def _looks_like_key_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(sig in msg for sig in _KEY_FAILURE_SIGNALS)

if not GEMINI_API_KEYS:
    # Don't crash the whole backend if the chatbot isn't configured yet —
    # /api/companies etc. should keep working even without a Gemini key.
    # ask_chatbot() below raises a clear error instead if this is missing.
    pass

SYSTEM_PROMPT = """You are SAIA, the AI assistant embedded in a stock-market
analytics platform (companies, price history, news sentiment, and ML price
predictions for a fixed universe of tickers).

Identity:
- You ARE "SAIA's AI agent" — that is your only identity in this
  conversation. Never describe yourself as "a large language model",
  "trained by Google", or similar generic self-descriptions, and never
  mention Gemini or any underlying model name. If asked who you are, briefly
  say you're SAIA's assistant for exploring the platform's stocks, news, and
  predictions.
- Stay in character and on topic even for small talk ("how are you",
  "what's up") — answer briefly and naturally, then steer toward how you can
  help with the platform's data, instead of giving a generic AI disclaimer.

Rules:
- Always use your tools to look up real numbers (prices, sentiment, tickers,
  predictions, watchlist contents). Never invent or estimate a price, a
  percentage change, or a company fact from memory — the platform's data can
  differ from what you were trained on.
- If a ticker/company can't be found via search_companies, say so plainly
  instead of guessing which company the user might mean.
- For questions about ranking/screening across the whole platform (highest
  price, biggest gainers/losers, most traded, largest market cap, etc.),
  use get_market_movers instead of saying you can't do it — you CAN answer
  these.
- For comparing two or more named tickers, use compare_tickers rather than
  calling get_company_details separately for each.
- For sector-level questions ("how's tech doing", "which sector is
  strongest"), use get_sector_overview.
- For general sentiment/mood questions about a ticker (not specific
  headlines), use get_ticker_sentiment_summary; use get_latest_news when
  they want actual articles.
- For a single named ticker's prediction, use get_ticker_prediction; use
  get_top_predictions only for "best N" lists.
- ML price predictions and news sentiment scores are model outputs, not
  guarantees. When you present them, briefly make clear they're
  model-generated and not financial advice.
- Keep answers concise and skimmable: short paragraphs, and bullet points or
  a small table when comparing several tickers or articles.
- Format lists as real Markdown: one "- " bullet per line, one fact per
  bullet. Never cram multiple fields onto one line separated by "*" (e.g.
  don't write "**TICKER**: * field1: x * field2: y" all on a single line) —
  give each field its own bullet/line so it renders as an actual list
  instead of raw asterisks.
- Reply in the same language the user wrote in. If they wrote Arabic, reply
  entirely in clear, natural Modern Standard Arabic (or mirror their dialect
  if they wrote in one, e.g. Egyptian Arabic) — well-formed sentences and
  correct grammar, not a stiff or literal translation. Never mix in English
  filler unless a term (like a ticker symbol) has no natural Arabic
  equivalent.
"""

# ---- CACHE for the (potentially huge — 10k+ rows) companies table --------
# search_companies/get_company_details filter this in Python, so we keep a
# short-lived cache instead of re-querying Databricks on every single tool
# call inside a multi-tool-call turn.
_companies_cache: dict = {"rows": None, "fetched_at": 0.0}
_COMPANIES_CACHE_TTL_SECONDS = 60


def _get_companies_cached() -> list:
    now = time.time()
    if (
        _companies_cache["rows"] is None
        or now - _companies_cache["fetched_at"] > _COMPANIES_CACHE_TTL_SECONDS
    ):
        _companies_cache["rows"] = _backend().get_companies()
        _companies_cache["fetched_at"] = now
    return _companies_cache["rows"]


# ---- TOOLS -----------------------------------------------------------------
# Each tool wraps existing backend logic (same MOCK_MODE / Databricks path
# the REST endpoints already use) and trims the result down to what's useful
# in an LLM context — the raw /api/companies response alone can be 10k+ rows.

@tool
def search_companies(query: str, limit: int = 10) -> list:
    """Search the platform's companies by ticker or company name (case-insensitive,
    partial match). Use this FIRST whenever the user names a company informally
    (e.g. "Apple", "tesla") to resolve it to an exact ticker before calling any
    other tool. Returns ticker, company_name, sector, industry, price,
    change_percent and market_cap for each match."""
    q = query.strip().lower()
    rows = _get_companies_cached()
    matches = [
        r
        for r in rows
        if q == (r.get("ticker") or "").lower()
        or q in (r.get("ticker") or "").lower()
        or q in (r.get("company_name") or "").lower()
        or q in (r.get("short_name") or "").lower()
    ]
    fields = ("ticker", "company_name", "sector", "industry", "price", "change_percent", "market_cap")
    return [{k: m.get(k) for k in fields} for m in matches[:limit]]


@tool
def get_company_details(ticker: str) -> dict:
    """Get the full snapshot for one exact ticker: sector, industry, exchange,
    country, market cap, P/E ratio, 52-week range, current price, day change,
    volume and average 3-month volume. Use search_companies first if you're not
    sure of the exact ticker."""
    rows = _get_companies_cached()
    ticker = ticker.strip().upper()
    for r in rows:
        if (r.get("ticker") or "").upper() == ticker:
            return r
    return {"error": f"No company found for ticker '{ticker}'. Try search_companies first."}


@tool
def get_stock_price_history(ticker: str, days: int = 30) -> list:
    """Get the last `days` trading days of OHLCV price history (open, high,
    low, close, adjusted close, volume) for one ticker, oldest first. Use this
    for trend questions ("how has X moved recently", "is X up or down this
    month")."""
    ticker = ticker.strip().upper()
    rows = _backend().get_stock_prices(ticker=ticker, days=days)
    if not rows:
        return [{"error": f"No price history found for ticker '{ticker}'."}]
    return rows


@tool
def get_latest_news(ticker: Optional[str] = None, limit: int = 5) -> list:
    """Get the most recent news articles, with FinBERT sentiment scores
    (positive_score / negative_score / neutral_score). Pass a ticker to filter
    to one company's news, or omit it for the latest market-wide news. Keep
    `limit` small (5-10) unless the user asks for more."""
    rows = _backend().get_news(ticker=ticker, limit=min(limit, 20), offset=0)
    fields = (
        "ticker", "title", "publisher_name", "source_url", "published_at",
        "positive_score", "negative_score", "neutral_score",
    )
    return [{k: r.get(k) for k in fields} for r in rows]


@tool
def get_top_predictions(limit: int = 10) -> list:
    """Get the tickers with the strongest ML-predicted 30-day price gains,
    sorted best-first. Each row has last_known_close_price,
    predicted_close_price and predicted_change_percent. These are model
    outputs, not guarantees — always caveat that when presenting them."""
    return _backend().get_predictions(limit=min(limit, 30))


@tool
def get_market_movers(sort_by: str = "price", direction: str = "desc", limit: int = 10) -> list:
    """Rank ALL companies on the platform by a numeric field and return the
    top N. Use this for ANY "which stock has the highest/lowest ___" or
    "top N stocks by ___" question across the whole universe — e.g. "highest
    priced stock today", "biggest gainers", "biggest losers", "most traded
    stocks" — instead of saying you can't do it.

    sort_by: one of "price" (current close price), "change_percent" (day's
        % move — use direction="desc" for top gainers, "asc" for top
        losers), "volume" (today's trading volume), or "market_cap".
    direction: "desc" for highest-first (default), "asc" for lowest-first.
    limit: how many to return (max 30).

    Returns ticker, company_name, and the sorted field's value for each,
    best-first. Rows missing that field are skipped rather than shown as
    None/zero.

    Note on change_percent: a $0.01 -> $0.06 move on a thinly-traded penny
    stock is a mathematically "real" +500% change_percent, but it's noise,
    not a meaningful market move — it drowns out genuine gainers/losers.
    So when sort_by="change_percent" this also drops stocks priced under
    $1 (last_known price too small for % change to be meaningful) and
    stocks with under 50k shares traded today (too illiquid for the price
    to reflect real market consensus)."""
    valid_fields = {"price", "change_percent", "volume", "market_cap"}
    if sort_by not in valid_fields:
        return [{"error": f"sort_by must be one of {sorted(valid_fields)}"}]
    rows = _get_companies_cached()
    ranked = [r for r in rows if r.get(sort_by) is not None]
    if sort_by == "change_percent":
        MIN_PRICE = 1.0
        MIN_VOLUME = 50_000
        ranked = [
            r for r in ranked
            if (r.get("price") or 0) >= MIN_PRICE and (r.get("volume") or 0) >= MIN_VOLUME
        ]
    ranked.sort(key=lambda r: r[sort_by], reverse=(direction != "asc"))
    fields = ("ticker", "company_name", sort_by)
    return [{k: r.get(k) for k in fields} for r in ranked[: min(limit, 30)]]


@tool
def compare_tickers(tickers: list) -> list:
    """Compare 2 or more tickers side by side: sector, industry, current
    price, day change_percent, market_cap, and (if available) the latest ML
    predicted_change_percent for each. Use this whenever the user asks to
    compare specific named tickers (e.g. "AAPL vs MSFT", "compare these
    three"). Unknown tickers are returned with an error note instead of
    being silently dropped."""
    companies = {r["ticker"]: r for r in _get_companies_cached() if r.get("ticker")}
    predictions = {}
    for t in tickers:
        t_upper = (t or "").strip().upper()
        if not t_upper:
            continue
        try:
            rows = _backend().get_predictions(limit=1, ticker=t_upper)
            if rows:
                predictions[t_upper] = rows[0]
        except Exception:
            pass
    out = []
    for t in tickers:
        t = (t or "").strip().upper()
        c = companies.get(t)
        if not c:
            out.append({"ticker": t, "error": "not found — try search_companies first"})
            continue
        pred = predictions.get(t)
        out.append({
            "ticker": t,
            "company_name": c.get("company_name"),
            "sector": c.get("sector"),
            "industry": c.get("industry"),
            "price": c.get("price"),
            "change_percent": c.get("change_percent"),
            "market_cap": c.get("market_cap"),
            "predicted_change_percent": pred.get("predicted_change_percent") if pred else None,
        })
    return out


@tool
def get_sector_overview(sector: Optional[str] = None) -> list:
    """Get performance rolled up by sector: average day change_percent,
    number of gainers/losers, total market cap, and company count. Pass a
    sector name to see just that sector's companies (ticker + price +
    change_percent), or omit it to compare ALL sectors against each other.
    Use this for questions like "how's the tech sector doing" or "which
    sector is up the most today"."""
    rows = _get_companies_cached()
    if sector:
        s = sector.strip().lower()
        matches = [r for r in rows if (r.get("sector") or "").lower() == s]
        if not matches:
            return [{"error": f"No companies found in sector '{sector}'."}]
        fields = ("ticker", "company_name", "price", "change_percent")
        return [{k: m.get(k) for k in fields} for m in matches]

    by_sector: dict = {}
    for r in rows:
        s = r.get("sector") or "Unknown"
        by_sector.setdefault(s, []).append(r)
    out = []
    for s, companies in by_sector.items():
        changes = [c["change_percent"] for c in companies if c.get("change_percent") is not None]
        caps = [c["market_cap"] for c in companies if c.get("market_cap") is not None]
        out.append({
            "sector": s,
            "company_count": len(companies),
            "avg_change_percent": (sum(changes) / len(changes)) if changes else None,
            "gainers": sum(1 for c in changes if c > 0),
            "losers": sum(1 for c in changes if c < 0),
            "total_market_cap": sum(caps) if caps else None,
        })
    out.sort(key=lambda r: (r["avg_change_percent"] is None, -(r["avg_change_percent"] or 0)))
    return out


@tool
def get_ticker_sentiment_summary(ticker: str, articles_checked: int = 20) -> dict:
    """Get an aggregated news-sentiment summary for one ticker (average
    FinBERT positive/negative/neutral scores across its most recent articles,
    plus an overall lean). Use this when the user asks about sentiment/mood
    around a stock in general, rather than wanting individual articles —
    use get_latest_news instead if they want actual headlines."""
    ticker = ticker.strip().upper()
    rows = _backend().get_news(ticker=ticker, limit=min(articles_checked, 50), offset=0)
    if not rows:
        return {"ticker": ticker, "error": "No recent news found for this ticker."}
    pos = [r["positive_score"] for r in rows if r.get("positive_score") is not None]
    neg = [r["negative_score"] for r in rows if r.get("negative_score") is not None]
    neu = [r["neutral_score"] for r in rows if r.get("neutral_score") is not None]
    avg_pos = sum(pos) / len(pos) if pos else 0
    avg_neg = sum(neg) / len(neg) if neg else 0
    avg_neu = sum(neu) / len(neu) if neu else 0
    lean = max(("positive", avg_pos), ("negative", avg_neg), ("neutral", avg_neu), key=lambda x: x[1])[0]
    return {
        "ticker": ticker,
        "articles_analyzed": len(rows),
        "avg_positive_score": avg_pos,
        "avg_negative_score": avg_neg,
        "avg_neutral_score": avg_neu,
        "overall_lean": lean,
    }


@tool
def get_ticker_prediction(ticker: str) -> dict:
    """Get the latest ML 30-day price prediction for ONE specific ticker
    (last_known_close_price, predicted_close_price, predicted_change_percent).
    Use this instead of get_top_predictions when the user names a specific
    ticker rather than asking for a top-N list. This is a model output, not
    a guarantee — always caveat that when presenting it."""
    ticker = ticker.strip().upper()
    rows = _backend().get_predictions(limit=1, ticker=ticker)
    if rows:
        return rows[0]
    return {"error": f"No prediction found for ticker '{ticker}'."}


@tool
def get_saved_articles(email: str) -> list:
    """Get the full details (title, ticker, sentiment, source) of the news
    articles this signed-in user has saved/bookmarked. Only call this if you
    actually have the user's email for this conversation."""
    saved = _backend().get_saved_news(email=email).get("saved_news", [])
    if not saved:
        return []
    rows = _backend().get_news_by_ids(article_ids=",".join(saved))
    fields = ("ticker", "title", "publisher_name", "published_at", "positive_score", "negative_score")
    return [{k: r.get(k) for k in fields} for r in rows]


@tool
def get_user_watchlist(email: str) -> dict:
    """Get the list of tickers a signed-in user (identified by email) has
    saved to their watchlist. Only call this if you actually have the user's
    email for this conversation."""
    return _backend().get_watchlist(email=email)


TOOLS = [
    search_companies,
    get_company_details,
    get_stock_price_history,
    get_latest_news,
    get_top_predictions,
    get_market_movers,
    compare_tickers,
    get_sector_overview,
    get_ticker_sentiment_summary,
    get_ticker_prediction,
    get_saved_articles,
    get_user_watchlist,
]

# ---- GRAPH -------------------------------------------------------------

# One lazily-built ChatGoogleGenerativeAI client per key, keyed by index
# into GEMINI_API_KEYS — built on first use, then reused.
_llm_clients: dict = {}

# Index of the key we currently believe is good. Starts at 0 and moves
# forward whenever the key at that index fails; future calls start from
# here instead of always retrying a key we already know is dead.
_current_key_idx = 0


def _get_llm_for_key(idx: int):
    if idx not in _llm_clients:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, api_key=GEMINI_API_KEYS[idx], temperature=0.3
        )
        _llm_clients[idx] = llm.bind_tools(TOOLS)
    return _llm_clients[idx]


def _call_model(state: MessagesState, config: RunnableConfig):
    global _current_key_idx

    if not GEMINI_API_KEYS:
        raise RuntimeError(
            "No Gemini API key is set — add GEMINI_API_KEYS (comma-separated for "
            "more than one) or GOOGLE_API_KEY/GEMINI_API_KEY to .env to enable the "
            "chatbot. Get one from https://aistudio.google.com/app/apikey"
        )

    # Keep only the most recent chunk of the conversation (by token budget)
    # before every model call — the checkpointer still holds the full
    # history, this just controls what actually gets sent to Gemini.
    # (Trimming needs a token_counter, which is just an LLM client — any
    # key's client counts tokens the same way, so this doesn't depend on
    # which key ends up actually serving the request below.)
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=_get_llm_for_key(_current_key_idx),
        max_tokens=6000,
        start_on="human",
        include_system=False,
    )

    # Durable facts about this user (name, preferences, ...) learned across
    # ALL of their conversations — passed in via config so they're available
    # even in a brand-new conversation_id/thread_id that has no message
    # history of its own yet. See ask_chatbot()'s `user_facts` param.
    user_facts = (config.get("configurable") or {}).get("user_facts") or []
    system_prompt = SYSTEM_PROMPT
    if user_facts:
        facts_block = "\n".join(f"- {fact}" for fact in user_facts)
        system_prompt += (
            "\n\nWhat you already know about this specific user, from earlier "
            "conversations (use it naturally, don't just recite it back):\n"
            f"{facts_block}"
        )
    messages = [SystemMessage(content=system_prompt)] + trimmed

    last_error = None
    num_keys = len(GEMINI_API_KEYS)
    # Try starting from the last known-good key, wrapping around through
    # every other key at most once each — so one dead key never causes an
    # infinite loop, and a request only ever fails once ALL keys have failed.
    for attempt in range(num_keys):
        idx = (_current_key_idx + attempt) % num_keys
        llm_with_tools = _get_llm_for_key(idx)
        try:
            response = llm_with_tools.invoke(messages)
            if idx != _current_key_idx:
                print(f"[chatbot] switched to Gemini key #{idx} ({_mask(GEMINI_API_KEYS[idx])}) "
                      f"after key #{_current_key_idx} failed")
            _current_key_idx = idx  # remember the key that actually worked
            return {"messages": [response]}
        except Exception as exc:
            last_error = exc
            if _looks_like_key_failure(exc):
                remaining = num_keys - attempt - 1
                print(
                    f"[chatbot] Gemini key #{idx} ({_mask(GEMINI_API_KEYS[idx])}) failed "
                    f"({exc}); {f'trying next key ({remaining} left)' if remaining else 'no more keys left'}"
                )
                continue
            # Not a key-related failure (bad request, tool bug, etc.) —
            # retrying with a different key won't help, so stop immediately.
            raise

    # Every key failed.
    raise RuntimeError(
        f"All {num_keys} configured Gemini API key(s) failed. Last error: {last_error}"
    ) from last_error


_graph_builder = StateGraph(MessagesState)
_graph_builder.add_node("agent", _call_model)
_graph_builder.add_node("tools", ToolNode(TOOLS))
_graph_builder.add_edge(START, "agent")
_graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
_graph_builder.add_edge("tools", "agent")

# Persistent (SQLite-backed) checkpointer — unlike the old MemorySaver, this
# survives `--reload` restarts, crashes, and redeploys, and works correctly
# even if uvicorn is ever run with multiple workers (each worker opens its
# own connection to the same file, so they all see the same conversation
# state). check_same_thread=False is required because FastAPI can call this
# from different threads than the one that opened the connection.
_sqlite_conn = sqlite3.connect(
    os.path.join(os.path.dirname(__file__), "saia_langgraph_checkpoints.db"),
    check_same_thread=False,
)
_checkpointer = SqliteSaver(_sqlite_conn)
# SqliteSaver doesn't create its own tables on construction — without this
# call, the very first checkpoint write (i.e. the first chat message ever
# sent after a fresh deploy / fresh checkpoints db) fails with
# "sqlite3.OperationalError: no such table: checkpoints". setup() is
# idempotent (safe to call every startup even once the tables exist), so
# it's fine to leave here permanently rather than as a one-off migration.
_checkpointer.setup()
chatbot_graph = _graph_builder.compile(checkpointer=_checkpointer)


def _extract_text(content) -> str:
    """Gemini's AIMessage.content isn't always a plain string — for
    responses that used "thinking", it comes back as a list of blocks, e.g.
    [{"type": "text", "text": "..."}, {"type": "thinking", "signature": ...}].
    Pull out just the actual reply text and ignore everything else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content)


def _seed_history_if_cold(config: dict, history: Optional[list]) -> None:
    """MemorySaver only keeps conversation state in this process's memory —
    it's empty again after every server restart/deploy, even though the
    persistent chat_messages table (and therefore the chat log the frontend
    shows the user) still has the full history. Without this, the model
    would silently "forget" everything from before the last restart while
    the user is staring at a chat log that says otherwise.

    `history` is the persisted log for this thread (oldest first, from
    app.py's chat_messages table), passed in by the caller. We only use it
    to seed the graph the FIRST time this thread_id is seen in this process
    (i.e. its checkpoint has no messages yet) — once the graph has its own
    state for the thread, that state is already the source of truth and we
    must not re-inject history on every turn (it would duplicate messages
    and grow unbounded)."""
    if not history:
        return
    try:
        existing = chatbot_graph.get_state(config)
        already_has_messages = bool(existing and existing.values.get("messages"))
    except Exception:
        already_has_messages = False
    if already_has_messages:
        return

    seed_messages = []
    for turn in history:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn.get("role") == "user":
            seed_messages.append(HumanMessage(content=content))
        else:
            seed_messages.append(AIMessage(content=content))
    if seed_messages:
        chatbot_graph.update_state(config, {"messages": seed_messages})


def ask_chatbot(
    message: str,
    thread_id: str = "default",
    history: Optional[list] = None,
    user_facts: Optional[list] = None,
) -> str:
    """Run one user turn through the graph and return Gemini's final text
    reply. `thread_id` scopes conversation memory — pass the user's email (or
    any stable per-session id) so follow-up questions keep context.

    `history`: optional persisted log for this thread — a list of
    {"role": "user"|"bot", "content": str} dicts, oldest first (the same
    shape /api/chat-history returns). Pass this whenever you have it (e.g.
    the caller just read it from a DB) so the agent still has real context
    the very first time it sees a thread after a server restart, instead of
    treating a mid-conversation message as if it were the opening line.

    `user_facts`: optional list of short durable facts about this user
    (name, stated preferences, ...) gathered across ALL of their
    conversations — not just this thread. Unlike `history`, this is passed
    on EVERY call (not just the first cold one), since it's what lets a
    brand-new conversation know things like the user's name."""
    config = {"configurable": {"thread_id": thread_id, "user_facts": user_facts or []}}
    _seed_history_if_cold(config, history)
    result = chatbot_graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return _extract_text(result["messages"][-1].content)


# ---- CROSS-CONVERSATION FACT EXTRACTION ---------------------------------
# A separate, tool-free, single-shot Gemini call (NOT part of the LangGraph
# chain above) that looks at one turn and decides whether it contains a
# durable fact worth remembering about the user in every future conversation
# (their name, a stated preference like "I only care about tech stocks",
# etc.) — as opposed to one-off, conversation-specific chat that shouldn't
# leak into unrelated future chats. Runs in the background from app.py after
# the reply has already been sent, so it never adds latency to the user.
FACTS_EXTRACTION_PROMPT = """You maintain a short list of durable facts about
one user of a stock-market analytics chatbot, based on what they say in chat.

Examples of facts WORTH keeping: their name, their profession, stocks/sectors
they say they care about, an explicit stated preference about how they want
answers formatted, their risk tolerance or investing goals.

Examples of facts NOT worth keeping: the specific ticker they happen to be
asking about right now, small talk, anything that only matters for the
current question and not for future unrelated conversations.

You will get the existing fact list (may be empty) and the latest user
message + assistant reply. Return the UPDATED full fact list as a JSON array
of short strings (each a standalone fact, e.g. "The user's name is Ahmed").
- If nothing new or worth remembering, return the existing list unchanged.
- If a new message contradicts or updates an old fact (e.g. a name
  correction), replace that fact rather than keeping both.
- Keep the list short (max ~10 facts) and each fact under ~15 words.
- Output ONLY the JSON array. No markdown fences, no commentary.
"""


def extract_user_facts(existing_facts: list, user_message: str, reply: str) -> list:
    """Best-effort: ask Gemini whether this turn taught us anything durable
    about the user, and return the (possibly unchanged) merged fact list.
    Falls back to `existing_facts` untouched on any error — a missed fact is
    fine, a crash here must never be allowed to bubble up to the user."""
    if not GEMINI_API_KEYS:
        return existing_facts
    import json as _json

    prompt = (
        f"Existing facts: {_json.dumps(existing_facts, ensure_ascii=False)}\n\n"
        f"User message: {user_message}\n\n"
        f"Assistant reply: {reply}\n\n"
        "Updated fact list (JSON array only):"
    )
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, api_key=GEMINI_API_KEYS[_current_key_idx], temperature=0
        )
        response = llm.invoke(
            [SystemMessage(content=FACTS_EXTRACTION_PROMPT), HumanMessage(content=prompt)]
        )
        text = _extract_text(response.content).strip()
        # Be lenient about accidental ```json fences even though we asked
        # the model not to include them.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        updated = _json.loads(text)
        if isinstance(updated, list) and all(isinstance(f, str) for f in updated):
            return updated[:10]
        return existing_facts
    except Exception as e:
        print(f"[chatbot] fact extraction failed: {e}")
        return existing_facts
