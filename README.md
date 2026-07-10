# Financial Analytics: From Data Exploration to Portfolio Optimization

A five-stage financial analytics project covering data exploration, time-series
forecasting, risk analysis, portfolio optimization, and interactive dashboard
design — built around a real, publicly-sourced five-stock portfolio.

**Portfolio:** AAPL (Technology) · MSFT (Technology) · JPM (Financials) · XOM (Energy) · JNJ (Healthcare)
**Period:** January 2013 – December 2017 (Weeks 1–2 use a longer AAPL-only window, Feb 2015–Feb 2017)
**Data sources:** publicly available, freely accessible historical equity price archives (see [Data Sources](#data-sources))

---

## Key results

| Stage | Headline finding |
|---|---|
| Data exploration | AAPL return/volume outliers cluster around earnings dates, not random noise — confirming they should be *kept*, not cleaned away, in later volatility estimates |
| Forecasting | ARIMA(1,1,1) and Holt's damped-trend smoothing perform almost identically (MAPE 6.43% vs. 6.50%) over a 30-day horizon; residuals show non-constant volatility |
| Risk & optimization | Reallocating from equal-weight to the max-Sharpe portfolio improves the Sharpe ratio from **1.01 → 1.26**, mainly by dropping XOM (near-zero return over the sample period) in favor of MSFT and JNJ |
| Dashboard | An interactive Plotly.js prototype makes the performance/risk/optimization trade-offs explorable in one interface instead of three static reports |
| Capstone | Synthesizes all four stages into one narrative and a set of concrete portfolio recommendations |

---

## Repository structure

```
.
├── requirements.txt
├── week1_data_exploration/
│   ├── eda_analysis.py          # data cleaning, outlier detection, EDA charts
│   └── report.docx              # full written report
├── week2_forecasting/
│   ├── forecasting_models.py    # stationarity tests, ARIMA, Holt's ES, diagnostics
│   └── report.docx
├── week3_risk_optimization/
│   ├── risk_analysis.py         # returns, volatility, beta, Value at Risk
│   ├── portfolio_optimization.py# Markowitz mean-variance optimization
│   ├── generate_charts.py       # all Week 3 report charts
│   └── report.docx
├── week4_dashboard/
│   ├── prepare_dashboard_data.py# exports Week 1-3 metrics to the dashboard
│   ├── dashboard/
│   │   ├── index.html           # the dashboard itself (Plotly.js, vanilla JS)
│   │   ├── data_inline.js        # generated data payload (inlined, see note below)
│   │   └── plotly.min.js         # local Plotly.js bundle (no CDN dependency)
│   ├── screenshots/              # captured dashboard states, used in report.docx
│   └── report.docx
└── week5_capstone/
    └── report.docx               # final synthesis report and recommendations
```

---

## Running the pipeline

Each week's script expects the outputs of the previous week to be present in
its working directory (or copied over — the scripts read plain CSV/JSON, no
database required). Install dependencies once from the repo root:

```bash
pip install -r requirements.txt
```

### Week 1 — Data exploration

```bash
cd week1_data_exploration
# download the source CSV (see Data Sources below) and save as raw_data.csv
python eda_analysis.py
# -> cleaned_data.csv, eda_report.json, charts/*.png
```

### Week 2 — Forecasting

```bash
cd week2_forecasting
cp ../week1_data_exploration/cleaned_data.csv .
python forecasting_models.py
# -> model_metrics.json, charts/*.png
```

### Week 3 — Risk analysis & portfolio optimization

```bash
cd week3_risk_optimization
# download the source CSV (see Data Sources below) and save as all_stocks_raw.csv
python risk_analysis.py           # -> portfolio_prices.csv, daily_returns.csv, risk_summary.csv
python portfolio_optimization.py  # -> efficient_frontier.csv, portfolio_results.json
python generate_charts.py         # -> charts/*.png
```

### Week 4 — Dashboard

```bash
cd week4_dashboard
cp ../week3_risk_optimization/{portfolio_prices,daily_returns,risk_summary}.csv .
cp ../week3_risk_optimization/efficient_frontier.csv .
cp ../week3_risk_optimization/portfolio_results.json .
python prepare_dashboard_data.py   # -> dashboard/data_inline.js
open dashboard/index.html          # or double-click it — no server needed
```

> **Note on `data_inline.js`:** the dashboard reads its data from an inlined
> JavaScript variable rather than `fetch()`-ing a JSON file. This is
> intentional — Chromium (and most browsers) block `fetch()` of local files
> when a page is opened directly via the `file://` protocol, so the data is
> baked into the page instead. A server-hosted version could switch back to
> a real `fetch('dashboard_data.json')` call.

### Week 5 — Capstone

`week5_capstone/report.docx` is a written synthesis of all four stages above;
it does not have its own script, since it draws entirely on the outputs and
findings already produced by Weeks 1–4.

---

## Data sources

- **AAPL daily OHLCV (Weeks 1–2):** [plotly/datasets](https://github.com/plotly/datasets) —
  `finance-charts-apple.csv`, a public republication of historical Yahoo Finance data.
- **Multi-stock DJIA-constituent daily prices (Weeks 3–4):** [szrlee/Stock-Time-Series-Analysis](https://github.com/szrlee/Stock-Time-Series-Analysis) —
  `all_stocks_2006-01-01_to_2018-01-01.csv`, a public archive of historical daily prices
  for 29 DJIA-constituent companies, from which the AAPL/MSFT/JPM/XOM/JNJ subset and
  an equal-weighted market benchmark (used for beta) were derived.

No paid data subscription or API key is required to reproduce any part of this project.

---

## Methodology references

- Box, G.E.P. & Jenkins, G.M. — *Time Series Analysis: Forecasting and Control* (ARIMA)
- Holt, C.C. (1957) — exponential smoothing with a trend component
- Markowitz, H. (1952) — "Portfolio Selection," *The Journal of Finance* (mean-variance optimization)
- J.P. Morgan RiskMetrics methodology — Value at Risk estimation

---

## Tech stack

Python (pandas, NumPy, statsmodels, SciPy, scikit-learn, Matplotlib, Seaborn) for all
analysis; Plotly.js + vanilla HTML/CSS/JavaScript for the dashboard prototype (no
backend framework required to run it — open `index.html` directly in a browser).
