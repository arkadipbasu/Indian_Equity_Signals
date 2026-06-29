"""
Contact: arkadipbasu.github.io
------
Streamlit dashboard: Indian Indices -- Signal-Weighted Market Outlook.
Run locally:   streamlit run app.py
"""

from __future__ import annotations
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st

from config import (
    INDEX_LIST, APP_TITLE, DISCLAIMER, CACHE_TTL_SECONDS, YF_TICKERS,
)
from logger_config import get_logger, log_event
import data_fetcher as df_mod
import sentiment as sent_mod
import signal_engine as eng
import predictor as pred_mod

logger = get_logger()

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")

# --------------------------------------------------------------------------
# Cached data-access wrappers (Streamlit-level caching; data_fetcher.py
# itself stays framework-agnostic so it can be unit tested standalone)
# --------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_nse_indices():
    return df_mod.fetch_nse_all_indices()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_global_signals():
    return df_mod.fetch_global_signals()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_fii_dii():
    return df_mod.fetch_fii_dii()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def cached_index_history(display_name: str):
    hist = df_mod.fetch_index_history(display_name)
    return hist  # DataFrame or None -- Streamlit can cache DataFrames fine


@st.cache_data(ttl=600, show_spinner=False)  # news/sentiment refresh slower
def cached_headlines(display_name: str):
    return df_mod.fetch_news_headlines(display_name)


def get_prev_close(display_name: str, nse_row, yf_hist) -> float | None:
    if nse_row and nse_row.get("prevClose"):
        try:
            return float(nse_row["prevClose"])
        except (TypeError, ValueError):
            pass
    if yf_hist is not None and len(yf_hist) >= 2:
        try:
            return float(yf_hist["Close"].iloc[-2])
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    st.subheader("arkadipbasu.github.io")
    if st.button("🔄 Refresh all data now", use_container_width=True):
        st.cache_data.clear()
        log_event("INFO", "manual_refresh_triggered")
        st.rerun()

    st.caption(f"Cache TTL: {CACHE_TTL_SECONDS}s for prices, 600s for news.")
    st.divider()
    st.subheader("Data sources")
    st.markdown(
        "- **NSE India** (allIndices, FII/DII) -- primary\n"
        "- **Yahoo Finance** -- global macro + cross-check\n"
        "- **Google News RSS + VADER** -- headline sentiment\n"
    )
    st.caption(
        "Want sharper news signal? Swap `data_fetcher.fetch_news_headlines` "
        "for a paid provider (e.g. Perigon, NewsAPI) -- same return shape."
    )
    st.divider()
    show_log = st.checkbox("Show recent log entries", value=False)
    st.divider()
    st.warning(DISCLAIMER)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("📈 " + APP_TITLE)
st.caption(
    f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST · "
    "Estimates refresh automatically every few minutes from cache."
)

# --------------------------------------------------------------------------
# Fetch shared data once
# --------------------------------------------------------------------------

with st.spinner("Pulling market signals..."):
    try:
        nse_rows = cached_nse_indices()
    except Exception:
        nse_rows = {}
        log_event("ERROR", "nse_rows_fetch_exception", trace=traceback.format_exc(limit=2))

    try:
        global_signals = cached_global_signals()
    except Exception:
        global_signals = {}
        log_event("ERROR", "global_signals_fetch_exception", trace=traceback.format_exc(limit=2))

    try:
        fii_dii = cached_fii_dii()
    except Exception:
        fii_dii = None

    global_norm = eng.compute_global_norm(global_signals)
    vix_level = global_norm.get("vix_level")

if not nse_rows:
    st.info(
        "NSE's live index feed didn't respond this refresh (it rate-limits "
        "scripted access at times) -- falling back to Yahoo Finance where "
        "available. Numbers may be stale; hit Refresh to retry NSE."
    )

# --------------------------------------------------------------------------
# Compute predictions for every index
# --------------------------------------------------------------------------

results = {}
for idx in INDEX_LIST:
    nse_row = df_mod.match_nse_index(idx, nse_rows) if nse_rows else None
    yf_hist = cached_index_history(idx)
    headlines = cached_headlines(idx)
    sentiment = sent_mod.score_headlines(headlines)

    signals = eng.compute_index_signals(idx, global_norm, nse_row, yf_hist, fii_dii, sentiment)
    composite_result = eng.compute_composite(idx, signals)
    prev_close = get_prev_close(idx, nse_row, yf_hist)

    prediction = pred_mod.predict_index(idx, prev_close, composite_result, vix_level)
    prediction["headlines"] = headlines
    prediction["sentiment"] = sentiment
    prediction["last_price"] = (nse_row.get("last") if nse_row else None)
    results[idx] = prediction

# --------------------------------------------------------------------------
# Overview table
# --------------------------------------------------------------------------

st.subheader("Market-wide snapshot")

badge = {"Higher": "🟢 Higher", "Lower": "🔴 Lower", "Flat": "🟡 Flat"}

overview_rows = []
for idx, r in results.items():
    overview_rows.append({
        "Index": idx,
        "Prev Close": r["prev_close"],
        "Predicted Open": r["predicted_open"],
        "Open vs Prev": badge.get(r["open_direction"], r["open_direction"]),
        "Predicted High": r["predicted_high"],
        "Predicted Low": r["predicted_low"],
        "Close vs Open": badge.get(r["close_vs_open_direction"], r["close_vs_open_direction"]),
        "Confidence": r["confidence"],
        "Composite": r["composite_score"],
    })

overview_df = pd.DataFrame(overview_rows).set_index("Index")
st.dataframe(overview_df, use_container_width=True)

bullish = sum(1 for r in results.values() if r["open_direction"] == "Higher")
bearish = sum(1 for r in results.values() if r["open_direction"] == "Lower")
c1, c2, c3 = st.columns(3)
c1.metric("Indexes leaning Higher at open", bullish)
c2.metric("Indexes leaning Lower at open", bearish)
c3.metric("India VIX", f"{vix_level:.2f}" if vix_level else "n/a")

st.divider()

# --------------------------------------------------------------------------
# Per-index detail tabs
# --------------------------------------------------------------------------

st.subheader("Per-index detail")
tabs = st.tabs(INDEX_LIST)

for tab, idx in zip(tabs, INDEX_LIST):
    r = results[idx]
    with tab:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Previous Close", f"{r['prev_close']:.2f}" if r["prev_close"] else "n/a")
        col2.metric(
            "Predicted Open",
            f"{r['predicted_open']:.2f}" if r["predicted_open"] else "n/a",
            f"{r['est_gap_pct']:+.2f}%",
        )
        col3.metric("Predicted High", f"{r['predicted_high']:.2f}" if r["predicted_high"] else "n/a")
        col4.metric("Predicted Low", f"{r['predicted_low']:.2f}" if r["predicted_low"] else "n/a")

        st.markdown(
            f"**Open vs previous close:** {badge.get(r['open_direction'], r['open_direction'])}  "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; **Close vs today's open:** "
            f"{badge.get(r['close_vs_open_direction'], r['close_vs_open_direction'])}  "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; **Confidence:** {r['confidence']}"
        )
        st.caption(
            f"Composite score {r['composite_score']:+.3f} (range -1 bearish to +1 bullish) · "
            f"signal agreement {r['agreement_ratio']*100:.0f}% · "
            f"data completeness {r['data_completeness']*100:.0f}% · "
            f"vol multiplier {r['vix_multiplier']}x"
        )

        with st.expander("Signal breakdown (what's driving this score)"):
            bdf = pd.DataFrame(r["breakdown"]).rename(columns={
                "signal": "Signal", "value": "Normalized value (-1..1)",
                "weight": "Weight", "contribution": "Contribution to score",
            })
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        with st.expander("Recent headlines used for sentiment"):
            if r["headlines"]:
                for h in r["headlines"][:8]:
                    st.markdown(f"- {h}")
                s = r["sentiment"]
                st.caption(f"VADER avg compound: {s['compound_avg']:+.3f} over {s['n']} headlines")
            else:
                st.caption("No headlines retrieved this refresh.")

# --------------------------------------------------------------------------
# Log viewer
# --------------------------------------------------------------------------

if show_log:
    st.divider()
    st.subheader("Recent log entries")
    try:
        with open("logs/app.log", "r", encoding="utf-8") as f:
            lines = f.readlines()[-150:]
        st.code("".join(lines) or "Log file is empty.", language="json")
    except FileNotFoundError:
        st.caption("No log file yet -- it's created on first run.")

st.divider()
st.caption(DISCLAIMER)
