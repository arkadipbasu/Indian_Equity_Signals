"""
predictor.py
------------
Converts a per-index composite signal score (-1..+1, from signal_engine)
into concrete, human-readable predictions:

  - Open  : Gap Up / Flat / Gap Down vs previous close, with an estimated
            gap percentage and implied open level.
  - High / Low : an estimated intraday range around the predicted open,
            skewed by the composite score and widened/narrowed by
            volatility (India VIX level).
  - Close : Higher / Lower / Flat vs the day's OPEN (not vs previous
            close) -- this captures the within-day follow-through (or
            fade) question the user asked for, using a mean-reversion
            haircut on very large gaps (large gaps statistically tend to
            partially fade intraday).

All magnitudes are ESTIMATES from a heuristic model, not point
forecasts -- they are presented as bands with a confidence label, never
as a guaranteed number.
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from config import VOL_PROFILE
from logger_config import logged, log_event


def clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


def direction_label(score: float, neutral_band: float = 0.12) -> str:
    if score > neutral_band:
        return "Higher"
    if score < -neutral_band:
        return "Lower"
    return "Flat"


def confidence_label(agreement_ratio: float, data_completeness: float, magnitude: float) -> str:
    """Blend of signal agreement, data availability, and conviction strength."""
    score = 0.45 * agreement_ratio + 0.30 * data_completeness + 0.25 * min(abs(magnitude) / 0.5, 1.0)
    if score >= 0.66:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


@logged
def predict_index(
    display_name: str,
    prev_close: Optional[float],
    composite_result: Dict[str, Any],
    vix_level: Optional[float],
) -> Dict[str, Any]:
    """
    Builds the full Open/High/Low/Close prediction block for one index.
    `composite_result` is the dict returned by signal_engine.compute_composite.
    """
    composite = composite_result["composite"]
    profile = VOL_PROFILE.get(display_name)

    # Volatility multiplier: India VIX around 13-14 is "normal"; scale
    # gap/range expectations up when fear is elevated, down when calm.
    # Clipped to a sane 0.7x - 1.6x band so a data gap doesn't blow this up.
    if vix_level:
        vix_mult = clip(1.0 + (vix_level - 14.0) / 30.0, 0.7, 1.6)
    else:
        vix_mult = 1.0

    est_gap_pct = composite * profile.base_gap_pct * vix_mult
    est_range_pct = profile.base_range_pct * vix_mult

    predicted_open = prev_close * (1 + est_gap_pct / 100) if prev_close else None

    # Skew the high/low band by the composite score: a bullish composite
    # implies more room above the open than below, and vice versa.
    skew = clip(composite, -0.6, 0.6)  # cap skew so bands never collapse
    upside_frac = est_range_pct * (0.55 + 0.35 * skew) / 100
    downside_frac = est_range_pct * (0.55 - 0.35 * skew) / 100

    predicted_high = predicted_open * (1 + upside_frac) if predicted_open else None
    predicted_low = predicted_open * (1 - downside_frac) if predicted_open else None

    # Close vs Open: momentum tends to partially carry through, but very
    # large gaps statistically show some intraday fade -- apply a mild
    # haircut (0.75x) on the composite when translating to a close call,
    # plus a small extra fade if the gap itself was already large.
    fade = 1.0
    if abs(est_gap_pct) > profile.base_gap_pct:  # bigger-than-typical gap
        fade = 0.85
    close_score = clip(composite * 0.75 * fade)

    open_dir = direction_label(composite)
    close_dir = direction_label(close_score)

    conf = confidence_label(
        composite_result.get("agreement_ratio", 0.0),
        composite_result.get("data_completeness", 0.0),
        composite,
    )

    result = {
        "index": display_name,
        "prev_close": prev_close,
        "composite_score": round(composite, 3),
        "open_direction": open_dir,
        "est_gap_pct": round(est_gap_pct, 3),
        "predicted_open": round(predicted_open, 2) if predicted_open else None,
        "predicted_high": round(predicted_high, 2) if predicted_high else None,
        "predicted_low": round(predicted_low, 2) if predicted_low else None,
        "close_vs_open_direction": close_dir,
        "close_vs_open_score": round(close_score, 3),
        "confidence": conf,
        "agreement_ratio": round(composite_result.get("agreement_ratio", 0.0), 2),
        "data_completeness": round(composite_result.get("data_completeness", 0.0), 2),
        "vix_multiplier": round(vix_mult, 2),
        "breakdown": composite_result.get("breakdown", []),
    }
    log_event(
        "INFO", "prediction_built", index=display_name,
        open_dir=open_dir, close_dir=close_dir, confidence=conf,
    )
    return result
