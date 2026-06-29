"""
signal_engine.py
-----------------
Turns raw, heterogeneous market data (percent changes, basis points,
crore flows, sentiment scores) into a common -1..+1 scale, then combines
them per-index using config.WEIGHTS into a single composite score plus a
transparent signal-by-signal breakdown (so the dashboard can show "why"
behind every number, not just a black-box verdict).
"""

from __future__ import annotations
from typing import Dict, Optional, Any
import math

import pandas as pd

from config import WEIGHTS, CRUDE_SIGN, SIGNAL_KEYS
from logger_config import logged, log_event


def clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize_pct(pct: Optional[float], scale: float) -> float:
    """Map a percent-change value onto -1..1 using `scale` as the percent
    move that should saturate the signal (e.g. scale=1.5 means a +-1.5%
    move maps to +-1.0)."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return 0.0
    return clip(pct / scale)


def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    if closes is None or len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(val)


@logged
def compute_global_norm(global_signals: Dict[str, Optional[Dict]]) -> Dict[str, float]:
    """Normalizes the macro bundle from data_fetcher.fetch_global_signals()."""

    def pct(key):
        d = global_signals.get(key)
        return d["pct_change"] if d else None

    us_components = [pct("sp500"), pct("nasdaq"), pct("dow")]
    us_components = [v for v in us_components if v is not None]
    global_us = normalize_pct(sum(us_components) / len(us_components), 1.5) if us_components else 0.0

    asia_components = [pct("nikkei"), pct("hangseng"), pct("shanghai"), pct("kospi")]
    asia_components = [v for v in asia_components if v is not None]
    asia = normalize_pct(sum(asia_components) / len(asia_components), 1.5) if asia_components else 0.0

    crude_pct = pct("brent_crude")
    crude_raw = normalize_pct(crude_pct, 3.0) if crude_pct is not None else 0.0

    dxy_pct = pct("dollar_index")
    dxy = -normalize_pct(dxy_pct, 1.0) if dxy_pct is not None else 0.0  # strong dollar -> EM headwind

    usdinr_pct = pct("usdinr")
    usdinr = -normalize_pct(usdinr_pct, 1.0) if usdinr_pct is not None else 0.0  # weak rupee -> headwind

    us10y = global_signals.get("us10y")
    us10y_signal = 0.0
    if us10y and us10y.get("pct_change") is not None:
        # ^TNX is already in yield points*10; treat the pct_change of the
        # yield level as a proxy for a yield-direction risk-off signal.
        us10y_signal = -normalize_pct(us10y["pct_change"], 4.0)

    vix = global_signals.get("india_vix")
    vix_chg = vix["pct_change"] if vix else None
    vix_signal = -normalize_pct(vix_chg, 8.0) if vix_chg is not None else 0.0

    return {
        "global_us": global_us,
        "asia": asia,
        "crude_raw": crude_raw,
        "dxy": dxy,
        "usdinr": usdinr,
        "us10y": us10y_signal,
        "vix_signal": vix_signal,
        "vix_level": vix["last"] if vix else None,
    }


@logged
def compute_index_signals(
    display_name: str,
    global_norm: Dict[str, float],
    nse_row: Optional[Dict],
    yf_hist: Optional[pd.DataFrame],
    fii_dii: Optional[Dict[str, float]],
    sentiment: Dict[str, float],
) -> Dict[str, float]:
    """Builds the full 10-signal vector (already on -1..1) for one index."""

    signals: Dict[str, float] = {
        "global_us": global_norm.get("global_us", 0.0),
        "asia": global_norm.get("asia", 0.0),
        "dxy": global_norm.get("dxy", 0.0),
        "usdinr": global_norm.get("usdinr", 0.0),
        "us10y": global_norm.get("us10y", 0.0),
        "vix_signal": global_norm.get("vix_signal", 0.0),
    }

    # Crude: shared magnitude, index-specific sign (config.CRUDE_SIGN)
    crude_raw = global_norm.get("crude_raw", 0.0)
    signals["crude"] = clip(crude_raw * CRUDE_SIGN.get(display_name, -1))

    # FII/DII net flow -> normalized by a typical +-3000cr single-day swing
    if fii_dii:
        net = fii_dii.get("fii_net_cr", 0.0) + fii_dii.get("dii_net_cr", 0.0)
        signals["fii_dii"] = normalize_pct(net, 3000.0)
    else:
        signals["fii_dii"] = 0.0

    # Momentum: prior day's % return + RSI deviation from neutral (50)
    momentum = 0.0
    if yf_hist is not None and len(yf_hist) >= 2:
        try:
            prev_ret = (yf_hist["Close"].iloc[-1] - yf_hist["Close"].iloc[-2]) / yf_hist["Close"].iloc[-2] * 100
            ret_signal = normalize_pct(prev_ret, 1.5)
            rsi = compute_rsi(yf_hist["Close"])
            rsi_signal = clip((rsi - 50) / 25) if rsi is not None else 0.0
            momentum = clip(0.6 * ret_signal + 0.4 * rsi_signal)
        except Exception:
            momentum = 0.0
    elif nse_row and nse_row.get("pChange") is not None:
        try:
            momentum = normalize_pct(float(nse_row["pChange"]), 1.5)
        except (TypeError, ValueError):
            momentum = 0.0
    signals["momentum"] = momentum

    signals["sentiment"] = clip(sentiment.get("compound_avg", 0.0))

    return signals


@logged
def compute_composite(display_name: str, signals: Dict[str, float]) -> Dict[str, Any]:
    """
    Weighted blend of `signals` for one index using config.WEIGHTS.
    Returns composite score, normalized weight breakdown (for display),
    and an "agreement_ratio" confidence proxy: the fraction of
    non-trivial signals (|s| > 0.1) that share the composite's sign.
    """
    weights = WEIGHTS.get(display_name, {})
    total_weight = sum(weights.get(k, 0.0) for k in SIGNAL_KEYS) or 1.0

    composite = 0.0
    breakdown = []
    for key in SIGNAL_KEYS:
        w = weights.get(key, 0.0) / total_weight
        s = signals.get(key, 0.0)
        contribution = w * s
        composite += contribution
        breakdown.append({
            "signal": key, "value": round(s, 3),
            "weight": round(w, 3), "contribution": round(contribution, 4),
        })
    composite = clip(composite)

    nontrivial = [b for b in breakdown if abs(b["value"]) > 0.1]
    if nontrivial:
        same_sign = sum(
            1 for b in nontrivial
            if (b["value"] > 0) == (composite > 0)
        )
        agreement_ratio = same_sign / len(nontrivial)
    else:
        agreement_ratio = 0.0

    data_completeness = sum(1 for b in breakdown if b["value"] != 0.0) / len(breakdown)

    log_event(
        "INFO", "composite_computed", index=display_name,
        composite=round(composite, 4), agreement=round(agreement_ratio, 2),
    )

    return {
        "composite": composite,
        "breakdown": breakdown,
        "agreement_ratio": agreement_ratio,
        "data_completeness": data_completeness,
    }
