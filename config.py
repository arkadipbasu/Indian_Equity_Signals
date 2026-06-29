"""
config.py
---------
Central configuration: index universe, data-source tickers, and the
per-index signal weight matrix used by the prediction engine.

NOTE ON METHODOLOGY
This app produces a HEURISTIC, RULE-BASED estimate built from a weighted
blend of well-known pre-market and intraday signals. It is NOT a trained
ML model, NOT investment advice, and carries no guarantee of accuracy.
It is meant as a structured research/decision-support aid, the same way
a trading desk reads a morning macro note before the bell.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List

# --------------------------------------------------------------------------
# 1. Index universe
# --------------------------------------------------------------------------

INDEX_LIST: List[str] = [
    "NIFTY 50",
    "NIFTY NEXT 50",
    "NIFTY BANK",
    "NIFTY DEFENCE",
    "NIFTY AUTO",
    "NIFTY ENERGY",
    "NIFTY 200",
    "NIFTY MIDCAP 150",
    "NIFTY SMLCAP 250",
]

# --------------------------------------------------------------------------
# 2. Data source tickers
# --------------------------------------------------------------------------

# Yahoo Finance tickers, where one reliably exists. NSE's own "All Indices"
# API (data_fetcher.fetch_nse_all_indices) is treated as the PRIMARY source
# for Indian index OHLC/prev-close because it covers every index below,
# including sectoral ones Yahoo doesn't carry. Yahoo is used as a
# secondary cross-check / fallback when NSE blocks the request.
YF_TICKERS: Dict[str, Optional[str]] = {
    "NIFTY 50": "^NSEI",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY DEFENCE": None,            # no reliable Yahoo symbol -> NSE only
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY ENERGY": "^CNXENERGY",     # best-effort; falls back to NSE if it 404s
    "NIFTY 200": "^CNX200",
    "NIFTY MIDCAP 150": "NIFTYMIDCAP150.NS",
    "NIFTY SMLCAP 250": "NIFTYSMLCAP250.NS",
}

# Keywords used to fuzzy-match the "index" field returned by NSE's
# allIndices API (NSE's naming is inconsistent: "NIFTY 50", "NIFTY IND
# DEFENCE", "NIFTY MIDCAP 150" etc.) -- first exact match wins, else the
# row whose name contains every keyword is used.
NSE_INDEX_KEYWORDS: Dict[str, List[str]] = {
    "NIFTY 50": ["NIFTY", "50"],
    "NIFTY NEXT 50": ["NIFTY", "NEXT", "50"],
    "NIFTY BANK": ["NIFTY", "BANK"],
    "NIFTY DEFENCE": ["DEFENCE"],
    "NIFTY AUTO": ["NIFTY", "AUTO"],
    "NIFTY ENERGY": ["NIFTY", "ENERGY"],
    "NIFTY 200": ["NIFTY", "200"],
    "NIFTY MIDCAP 150": ["MIDCAP", "150"],
    "NIFTY SMLCAP 250": ["SMLCAP", "250"],
}

# Global / macro tickers (all on Yahoo Finance)
GLOBAL_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "nikkei": "^N225",
    "hangseng": "^HSI",
    "shanghai": "000001.SS",
    "kospi": "^KS11",
    "brent_crude": "BZ=F",
    "dollar_index": "DX-Y.NYB",
    "usdinr": "INR=X",
    "us10y": "^TNX",
    "india_vix": "^INDIAVIX",
}

# Google News RSS search terms used for headline sentiment, per index.
NEWS_QUERY: Dict[str, str] = {
    "NIFTY 50": "Nifty 50 OR Sensex market",
    "NIFTY NEXT 50": "Nifty Next 50 midcap large stocks",
    "NIFTY BANK": "Bank Nifty RBI banking stocks India",
    "NIFTY DEFENCE": "India defence stocks order export",
    "NIFTY AUTO": "India auto stocks sales EV",
    "NIFTY ENERGY": "India energy oil gas power stocks",
    "NIFTY 200": "Indian stock market broad",
    "NIFTY MIDCAP 150": "midcap stocks India",
    "NIFTY SMLCAP 250": "smallcap stocks India",
}

# --------------------------------------------------------------------------
# 3. Signal weight matrix
# --------------------------------------------------------------------------
# Each index gets a weight (0-1, need not sum to 1 -- normalized at runtime)
# across these signal components:
#   global_us   : overnight US market direction
#   asia        : same-session Asian market direction
#   crude       : Brent crude % change (sign flipped per-index below)
#   dxy         : Dollar Index % change (inverse to INR assets)
#   usdinr      : USD/INR % change (inverse; rupee weakness = headwind)
#   us10y       : US 10Y yield change in bps (inverse; risk-off signal)
#   vix_signal  : India VIX % change (inverse; rising fear = bearish)
#   fii_dii     : previous day's net FII+DII cash flow
#   momentum    : prior-day return + short-term RSI deviation
#   sentiment   : VADER sentiment of recent headlines for that index

SIGNAL_KEYS: List[str] = [
    "global_us", "asia", "crude", "dxy", "usdinr",
    "us10y", "vix_signal", "fii_dii", "momentum", "sentiment",
]

WEIGHTS: Dict[str, Dict[str, float]] = {
    "NIFTY 50": {
        "global_us": 0.20, "asia": 0.10, "crude": 0.10, "dxy": 0.08,
        "usdinr": 0.07, "us10y": 0.08, "vix_signal": 0.10,
        "fii_dii": 0.12, "momentum": 0.10, "sentiment": 0.05,
    },
    "NIFTY NEXT 50": {
        "global_us": 0.16, "asia": 0.08, "crude": 0.08, "dxy": 0.07,
        "usdinr": 0.06, "us10y": 0.07, "vix_signal": 0.11,
        "fii_dii": 0.18, "momentum": 0.14, "sentiment": 0.05,
    },
    "NIFTY BANK": {
        "global_us": 0.18, "asia": 0.07, "crude": 0.03, "dxy": 0.08,
        "usdinr": 0.07, "us10y": 0.17, "vix_signal": 0.10,
        "fii_dii": 0.20, "momentum": 0.07, "sentiment": 0.03,
    },
    "NIFTY DEFENCE": {
        "global_us": 0.08, "asia": 0.04, "crude": 0.04, "dxy": 0.04,
        "usdinr": 0.04, "us10y": 0.04, "vix_signal": 0.08,
        "fii_dii": 0.14, "momentum": 0.20, "sentiment": 0.30,
    },
    "NIFTY AUTO": {
        "global_us": 0.14, "asia": 0.06, "crude": 0.20, "dxy": 0.06,
        "usdinr": 0.14, "us10y": 0.05, "vix_signal": 0.08,
        "fii_dii": 0.12, "momentum": 0.10, "sentiment": 0.05,
    },
    "NIFTY ENERGY": {
        "global_us": 0.12, "asia": 0.06, "crude": 0.30, "dxy": 0.05,
        "usdinr": 0.08, "us10y": 0.04, "vix_signal": 0.07,
        "fii_dii": 0.10, "momentum": 0.13, "sentiment": 0.05,
    },
    "NIFTY 200": {
        "global_us": 0.18, "asia": 0.09, "crude": 0.09, "dxy": 0.07,
        "usdinr": 0.07, "us10y": 0.07, "vix_signal": 0.10,
        "fii_dii": 0.14, "momentum": 0.12, "sentiment": 0.07,
    },
    "NIFTY MIDCAP 150": {
        "global_us": 0.10, "asia": 0.05, "crude": 0.06, "dxy": 0.05,
        "usdinr": 0.05, "us10y": 0.05, "vix_signal": 0.14,
        "fii_dii": 0.22, "momentum": 0.20, "sentiment": 0.08,
    },
    "NIFTY SMLCAP 250": {
        "global_us": 0.07, "asia": 0.04, "crude": 0.05, "dxy": 0.04,
        "usdinr": 0.04, "us10y": 0.04, "vix_signal": 0.18,
        "fii_dii": 0.22, "momentum": 0.22, "sentiment": 0.10,
    },
}

# Sign of the crude-oil signal per index: most of the market treats
# higher crude as a cost/inflation headwind (-1). NIFTY ENERGY is tilted
# upstream (ONGC/Oil India type names) so higher crude is modestly
# constructive (+1) for the index as a whole; this is a simplifying
# assumption, not a precise refiner-vs-producer decomposition.
CRUDE_SIGN: Dict[str, int] = {idx: -1 for idx in INDEX_LIST}
CRUDE_SIGN["NIFTY ENERGY"] = 1

# --------------------------------------------------------------------------
# 4. Volatility / range model parameters per index
# --------------------------------------------------------------------------
# base_gap_pct   : typical overnight gap magnitude at composite score = 1.0
# base_range_pct : typical full day high-low range as % of open

@dataclass
class VolProfile:
    base_gap_pct: float
    base_range_pct: float


VOL_PROFILE: Dict[str, VolProfile] = {
    "NIFTY 50":         VolProfile(0.35, 0.80),
    "NIFTY NEXT 50":    VolProfile(0.45, 1.00),
    "NIFTY BANK":       VolProfile(0.45, 1.10),
    "NIFTY DEFENCE":    VolProfile(0.70, 1.80),
    "NIFTY AUTO":       VolProfile(0.45, 1.00),
    "NIFTY ENERGY":     VolProfile(0.45, 1.05),
    "NIFTY 200":        VolProfile(0.35, 0.85),
    "NIFTY MIDCAP 150": VolProfile(0.55, 1.30),
    "NIFTY SMLCAP 250": VolProfile(0.65, 1.55),
}

# --------------------------------------------------------------------------
# 5. Misc app constants
# --------------------------------------------------------------------------

CACHE_TTL_SECONDS = 120
REQUEST_TIMEOUT = 8
LOG_DIR = "logs"
APP_TITLE = "Indian Indices -- Signal-Weighted Market Outlook"

DISCLAIMER = (
    "Educational / research tool only. This dashboard combines public "
    "market signals into a heuristic, weighted estimate of index direction. "
    "It is NOT investment advice, NOT a guarantee of outcome, and should "
    "not be the sole basis for any trading decision. Markets can and do "
    "move against any model. The author/operator is not a registered "
    "investment advisor."
)
