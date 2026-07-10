"""
Week 4 — Dashboard data preparation
Exports the metrics computed in Weeks 1-3 into a single JSON payload
consumed by dashboard/index.html (inlined as dashboard/data_inline.js
to avoid file:// fetch/CORS restrictions when opening the dashboard
directly from disk).

Run:
    pip install pandas numpy statsmodels
    python prepare_dashboard_data.py
(expects portfolio_prices.csv, daily_returns.csv, risk_summary.csv,
 efficient_frontier.csv, portfolio_results.json from week3_risk_optimization)
"""

import json
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def build_payload():
    prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
    returns = pd.read_csv("daily_returns.csv", index_col="Date", parse_dates=True)
    risk = pd.read_csv("risk_summary.csv", index_col=0)
    frontier = pd.read_csv("efficient_frontier.csv")
    with open("portfolio_results.json") as f:
        port_results = json.load(f)

    prices_w = prices.resample("W").last().dropna()

    data = {
        "dates": prices_w.index.strftime("%Y-%m-%d").tolist(),
        "prices": {t: prices_w[t].round(2).tolist() for t in prices_w.columns},
        "tickers": prices_w.columns.tolist(),
        "risk": {
            t: {
                "ann_return": round(risk.loc[t, "Ann_Return"] * 100, 2),
                "ann_vol": round(risk.loc[t, "Ann_Volatility"] * 100, 2),
                "beta": round(risk.loc[t, "Beta"], 2),
                "var95": round(risk.loc[t, "Hist_VaR_95"] * 100, 2),
                "var99": round(risk.loc[t, "Hist_VaR_99"] * 100, 2),
            }
            for t in risk.index
        },
        "frontier": {
            "vol": [round(v * 100, 2) for v in frontier["Volatility"].tolist()],
            "ret": [round(r * 100, 2) for r in frontier["Return"].tolist()],
        },
        "portfolios": {
            name: {
                "weights": {k: round(v * 100, 1) for k, v in val["weights"].items()},
                "return": round(val["return"] * 100, 2),
                "vol": round(val["vol"] * 100, 2),
                "sharpe": round(val["sharpe"], 2),
                "var95": round(val["var95_1day"] * 100, 2),
            }
            for name, val in port_results.items()
        },
        "correlation": returns.corr().round(2).to_dict(),
    }

    # 12-week forecast of the max-Sharpe-weighted notional portfolio value
    tickers = prices.columns.tolist()
    w_sharpe = np.array([port_results["Max_Sharpe"]["weights"][t] for t in tickers])
    port_value = (prices * w_sharpe).sum(axis=1)
    port_value_w = port_value.resample("W").last().dropna()

    fit = ExponentialSmoothing(port_value_w, trend="add", damped_trend=True).fit()
    fc_steps = 12
    forecast = fit.forecast(fc_steps)
    resid_std = fit.resid.std()
    ci_upper = forecast + 1.96 * resid_std * np.sqrt(np.arange(1, fc_steps + 1))
    ci_lower = forecast - 1.96 * resid_std * np.sqrt(np.arange(1, fc_steps + 1))

    data["forecast"] = {
        "hist_dates": port_value_w.index.strftime("%Y-%m-%d").tolist()[-52:],
        "hist_values": port_value_w.round(2).tolist()[-52:],
        "fc_dates": forecast.index.strftime("%Y-%m-%d").tolist(),
        "fc_values": forecast.round(2).tolist(),
        "ci_upper": ci_upper.round(2).tolist(),
        "ci_lower": ci_lower.round(2).tolist(),
    }

    return data


if __name__ == "__main__":
    data = build_payload()

    with open("dashboard/dashboard_data.json", "w") as f:
        json.dump(data, f)

    # Also write as an inlined JS variable — the dashboard reads this directly
    # rather than fetch()-ing the JSON, since Chromium blocks fetch() of local
    # files opened via the file:// protocol.
    with open("dashboard/data_inline.js", "w") as f:
        f.write("const DATA_INLINE = " + json.dumps(data) + ";")

    print("Dashboard data written to dashboard/dashboard_data.json and dashboard/data_inline.js")
