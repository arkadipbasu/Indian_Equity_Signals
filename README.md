# Indian Indices — Signal-Weighted Market Outlook

A Streamlit dashboard that estimates today's likely **Open / High / Low**
(vs previous close) and **Close direction** (vs today's open) for nine
NSE indices, by combining free public market signals into a transparent,
weighted score per index:

- NIFTY 50, NIFTY NEXT 50, NIFTY BANK, NIFTY DEFENCE, NIFTY AUTO,
  NIFTY ENERGY, NIFTY 200, NIFTY MIDCAP 150, NIFTY SMLCAP 250

> **This is a heuristic research tool, not a trained ML model and not
> investment advice.** See the in-app disclaimer. Every prediction shows
> its full signal breakdown so you can judge it yourself rather than
> trust a black box.

---

## How it works

| Stage | File | What it does |
|---|---|---|
| 1. Fetch | `data_fetcher.py` | Pulls NSE's free `allIndices` + FII/DII endpoints, Yahoo Finance global macro (US/Asia indices, Brent crude, DXY, USDINR, US 10Y, India VIX), and Google News RSS headlines. |
| 2. Score | `sentiment.py` | VADER (offline, free) sentiment on headlines. |
| 3. Normalize & weight | `signal_engine.py` | Converts every raw signal to a common -1..+1 scale, then blends with per-index weights from `config.py` (e.g. NIFTY BANK weighs US yields & FII/DII heavily; NIFTY DEFENCE weighs domestic news/momentum heavily; NIFTY ENERGY weighs crude oil heavily). |
| 4. Predict | `predictor.py` | Turns the composite score + India VIX level into an estimated gap %, predicted Open/High/Low, and a Close-vs-Open call, each tagged with a High/Medium/Low confidence label. |
| 5. Display | `app.py` | Streamlit dashboard: market-wide table + per-index tabs with full signal breakdown and headlines. |
| Logging | `logger_config.py` | Every fetch/compute step writes a structured JSON line to `logs/app.log` (rotated daily) via a `@logged` decorator, plus console output. |

### Signals used (per index, weights vary)

`global_us`, `asia`, `crude` (sign flipped for NIFTY ENERGY), `dxy`,
`usdinr`, `us10y`, `vix_signal`, `fii_dii`, `momentum` (prior-day return +
RSI), `sentiment` (headline VADER score).

### Upgrading the news signal

The default news source is Google News RSS (free, no key). If you have a
paid news-intelligence API key — **Perigon**, NewsAPI, etc. — swap the
body of `data_fetcher.fetch_news_headlines()` to call that provider
instead; it just needs to keep returning a `list[str]` of headlines, so
nothing downstream (`sentiment.py`, `app.py`) needs to change.

---

## Local setup (macOS / Linux)

```bash
# 1. Clone / unzip this project, then cd into it
cd market_signal_dashboard

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows (PowerShell), if needed

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Logs accumulate in
`logs/app.log` (one JSON object per line, daily rotation, 14-day
retention). Toggle "Show recent log entries" in the sidebar to view the
last 150 lines in-app without leaving the browser.

### Notes on NSE access

`nseindia.com`'s public API occasionally rate-limits or blocks scripted
requests (it's an undocumented endpoint, not an official API). The app
handles this gracefully — it logs a warning, shows an in-app notice, and
falls back to Yahoo Finance data where a ticker exists. Hitting
**Refresh** in the sidebar after a minute usually succeeds.

---

## Deploying to Streamlit Community Cloud

1. Push this project to a **public or private GitHub repo**, keeping the
   structure as-is (`app.py` at the repo root, `requirements.txt`,
   `.streamlit/config.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **"New app"**.
3. Pick the repo/branch, set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. No secrets/API keys are required for the default (free) data sources.
   If you add a paid news API key later, set it under **App settings →
   Secrets** as `PERIGON_API_KEY = "..."` and read it in
   `data_fetcher.py` via `st.secrets["PERIGON_API_KEY"]`.

**Caveat for cloud deployments:** Streamlit Cloud's outbound IPs are
shared infrastructure, so NSE may rate-limit it more aggressively than
your home IP. The Yahoo Finance fallback path covers the indices that
have a Yahoo ticker (`config.YF_TICKERS`) if that happens; NIFTY
DEFENCE has no reliable Yahoo symbol today, so it depends on NSE access
succeeding.

---

## Project layout

```
market_signal_dashboard/
├── app.py              # Streamlit UI (entry point)
├── config.py            # Index list, tickers, weight matrix, vol profiles
├── data_fetcher.py       # NSE / Yahoo Finance / Google News RSS calls
├── sentiment.py          # VADER headline sentiment
├── signal_engine.py       # Normalization + weighted composite scoring
├── predictor.py           # Composite score -> Open/High/Low/Close calls
├── logger_config.py       # JSON logging + @logged decorator
├── requirements.txt
├── .streamlit/config.toml # Dark theme + server config
├── .gitignore
└── logs/                  # Created automatically at runtime
```

## Extending it

- **Backtest the weights**: log predictions daily (`logs/app.log` already
  has everything) and compare to actual next-day OHLC to tune
  `config.WEIGHTS` and `config.VOL_PROFILE` empirically over time.
- **Add GIFT Nifty premium**: the single most predictive pre-market
  signal for NIFTY 50's open is the GIFT Nifty premium/discount to the
  previous cash close. There's no free, reliable API for it today; if
  you get access to one, add it as a new signal key in
  `config.SIGNAL_KEYS` / `WEIGHTS` and feed it into
  `signal_engine.compute_index_signals`.
- **Swap in a real model**: once you've logged enough daily
  signal/outcome pairs, `signals` dict per index is already a clean
  feature vector — a logistic regression or gradient-boosted classifier
  could replace the linear weighted-sum in `compute_composite` with
  minimal refactoring.
