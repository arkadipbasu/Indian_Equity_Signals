"""
sentiment.py
------------
Lightweight, fully offline sentiment scoring for headlines using VADER
(Valence Aware Dictionary and sEntiment Reasoner) -- free, open-source,
no API key, works well on short news-style text. This keeps the
"sentiment" signal in the model functional out-of-the-box; if you have a
paid news-intelligence API key (e.g. Perigon, NewsAPI) you can swap in
its own sentiment/relevance scores here without touching the rest of
the pipeline.
"""

from __future__ import annotations
from typing import List, Dict

from logger_config import logged, log_event

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _analyzer = None


@logged
def score_headlines(headlines: List[str]) -> Dict[str, float]:
    """
    Returns {'compound_avg': -1..1, 'n': count, 'positive': n, 'negative': n}.
    compound_avg of 0.0 with n=0 means "no data" (caller should treat the
    sentiment signal as neutral/low-confidence in that case).
    """
    if not headlines or _analyzer is None:
        return {"compound_avg": 0.0, "n": 0, "positive": 0, "negative": 0}

    scores = []
    pos = neg = 0
    for h in headlines:
        try:
            s = _analyzer.polarity_scores(h)["compound"]
        except Exception:
            continue
        scores.append(s)
        if s > 0.15:
            pos += 1
        elif s < -0.15:
            neg += 1

    if not scores:
        return {"compound_avg": 0.0, "n": 0, "positive": 0, "negative": 0}

    avg = sum(scores) / len(scores)
    log_event("INFO", "sentiment_scored", n=len(scores), avg=round(avg, 3))
    return {
        "compound_avg": round(avg, 4),
        "n": len(scores),
        "positive": pos,
        "negative": neg,
    }
