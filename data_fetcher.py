"""
data_fetcher.py
----------------
All outbound data retrieval lives here, isolated from Streamlit so it can
be unit-tested / run from a plain script. Every function is defensive:
network calls are wrapped in try/except, return a consistent dict shape,
and never raise -- callers get partial data + an "ok" flag rather than a
crashed dashboard. Every public fetch function is decorated with
@logged so every call is auditable in logs/app.log.

Data sources used (all free, no API key required):
  - Yahoo Finance (via yfinance)            -> global macro + cross-check
  - NSE India "All Indices" public endpoint -> primary Indian index OHLC
  - NSE India FII/DII provisional endpoint  -> previous day flows
  - Google News RSS                         -> headlines for sentiment

NOTE: NSE's website actively rate-limits / blocks scripted access at
times. Every NSE call therefore degrades gracefully to "unavailable"
rather than failing the whole app -- yfinance cross-check or cached
values fill the gap where possible.

Optional paid upgrade path: if the user has a Perigon (or NewsAPI, etc.)
API key, swap fetch_news_headlines()'s Google-News-RSS call for that
provider's endpoint -- the rest of the pipeline (sentiment.py) only
needs a list of headline strings back, so the swap is local to this
one function.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any

import requests
import feedparser
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from config import (
    GLOBAL_TICKERS, YF_TICKERS, NSE_INDEX_KEYWORDS, NEWS_QUERY,
    REQUEST_TIMEOUT,
)
from logger_config import logged, log_event

NSE_BASE = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}


# --------------------------------------------------------------------------
# NSE: All-indices snapshot (primary source for Indian index OHLC)
# --------------------------------------------------------------------------

@logged
def fetch_nse_all_indices() -> Dict[str, Dict[str, Any]]:
    """
    Hits NSE's public allIndices endpoint and returns a dict keyed by the
    raw NSE index name -> {last, open, high, low, prevClose, pChange}.
    Returns {} (not an exception) if NSE blocks/times out.
    """
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        # "warm up" the session -- NSE requires a prior page hit to set
        # cookies before the API call will succeed.
        session.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
        time.sleep(0.4)
        resp = session.get(
            f"{NSE_BASE}/api/allIndices", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data", [])
        out = {}
        for row in rows:
            name = row.get("index") or row.get("indexName")
            if not name:
                continue
            out[name.strip().upper()] = {
                "last": row.get("last"),
                "open": row.get("open"),
                "high": row.get("high") or row.get("dayHigh"),
                "low": row.get("low") or row.get("dayLow"),
                "prevClose": row.get("previousClose"),
                "pChange": row.get("percentChange") or row.get("perChange"),
            }
        log_event("INFO", "nse_all_indices_ok", rows=len(out))
        return out
    except Exception as exc:
        log_event("WARNING", "nse_all_indices_failed", error=str(exc))
        return {}


def match_nse_index(display_name: str, nse_rows: Dict[str, Dict]) -> Optional[Dict]:
    """Fuzzy-match our display name against NSE's raw index name keys."""
    target = display_name.strip().upper()
    if target in nse_rows:
        return nse_rows[target]
    keywords = NSE_INDEX_KEYWORDS.get(display_name, target.split())
    best, best_score = None, 0
    for raw_name, row in nse_rows.items():
        score = sum(1 for kw in keywords if kw in raw_name)
        if score == len(keywords) and score > best_score:
            best, best_score = row, score
    return best


# --------------------------------------------------------------------------
# NSE: FII/DII provisional cash flow (previous trading day)
# --------------------------------------------------------------------------

@logged
def fetch_fii_dii() -> Optional[Dict[str, float]]:
    """Returns {'fii_net_cr': x, 'dii_net_cr': y} in INR crore, or None."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        session.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
        time.sleep(0.3)
        resp = session.get(
            f"{NSE_BASE}/api/fiidiiTradeReact", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        out = {}
        for row in data:
            cat = (row.get("category") or "").upper()
            net = row.get("netValue") or row.get("buyValue", 0)
            try:
                net = float(net)
            except (TypeError, ValueError):
                continue
            if "FII" in cat or "FPI" in cat:
                out["fii_net_cr"] = net
            elif "DII" in cat:
                out["dii_net_cr"] = net
        if out:
            log_event("INFO", "fii_dii_ok", **out)
            return out
        return None
    except Exception as exc:
        log_event("WARNING", "fii_dii_failed", error=str(exc))
        return None


# --------------------------------------------------------------------------
# Yahoo Finance: global macro signals + per-index cross-check
# --------------------------------------------------------------------------

@logged
def fetch_yf_change_pct(ticker: str) -> Optional[Dict[str, float]]:
    """Returns {'last': x, 'prev_close': y, 'pct_change': z} for one ticker."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")
        if hist is None or hist.empty or len(hist) < 2:
            return None
        last_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        pct = (last_close - prev_close) / prev_close * 100.0
        return {"last": last_close, "prev_close": prev_close, "pct_change": pct}
    except Exception as exc:
        log_event("WARNING", "yf_fetch_failed", ticker=ticker, error=str(exc))
        return None


@logged
def fetch_global_signals() -> Dict[str, Optional[Dict[str, float]]]:
    """One bundled call for every macro ticker in config.GLOBAL_TICKERS."""
    out = {}
    for key, ticker in GLOBAL_TICKERS.items():
        out[key] = fetch_yf_change_pct(ticker)
    return out


@logged
def fetch_index_history(display_name: str, period: str = "1mo") -> Optional[pd.DataFrame]:
    """Daily OHLC history for momentum/RSI calc, via the YF ticker if one exists."""
    ticker = YF_TICKERS.get(display_name)
    if ticker is None or yf is None:
        return None
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if hist is None or hist.empty:
            return None
        return hist
    except Exception as exc:
        log_event("WARNING", "yf_history_failed", index=display_name, error=str(exc))
        return None


# --------------------------------------------------------------------------
# News headlines (Google News RSS -- free, no API key)
# --------------------------------------------------------------------------

@logged
def fetch_news_headlines(display_name: str, max_items: int = 12) -> List[str]:
    """
    Pulls recent headlines for an index's configured search query via
    Google News RSS. Returns a plain list of headline strings (titles
    only -- no article bodies are fetched or stored, keeping this
    lightweight and copyright-safe).
    """
    query = NEWS_QUERY.get(display_name, display_name)
    url = (
        "https://news.google.com/rss/search?q="
        + requests.utils.quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    try:
        feed = feedparser.parse(url)
        titles = [entry.title for entry in feed.entries[:max_items] if getattr(entry, "title", None)]
        log_event("INFO", "news_fetch_ok", index=display_name, count=len(titles))
        return titles
    except Exception as exc:
        log_event("WARNING", "news_fetch_failed", index=display_name, error=str(exc))
        return []
