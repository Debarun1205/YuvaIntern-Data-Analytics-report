"""
Week 3 — Risk Analysis
Case study: 5-asset equity portfolio (AAPL, MSFT, JPM, XOM, JNJ), Jan 2013 - Dec 2017

Computes annualized return/volatility, beta vs. an equal-weighted market
benchmark, and historical + parametric Value at Risk for each holding.

Source data: szrlee/Stock-Time-Series-Analysis (GitHub) —
https://raw.githubusercontent.com/szrlee/Stock-Time-Series-Analysis/master/data/all_stocks_2006-01-01_to_2018-01-01.csv

Run:
    pip install pandas numpy scipy
    python risk_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
TRADING_DAYS = 252


def load_data(path="all_stocks_raw.csv", start="2013-01-01"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"] >= start]

    prices = df[df["Name"].isin(TICKERS)].pivot(index="Date", columns="Name", values="Close")
    prices = prices.sort_index().ffill().dropna()

    # Equal-weighted benchmark built from every ticker in the source dataset
    all_close = df.pivot(index="Date", columns="Name", values="Close").sort_index()
    market = all_close.loc[prices.index].mean(axis=1)

    return prices, market


def compute_risk_metrics(prices, market):
    returns = prices.pct_change().dropna()
    market_ret = market.pct_change().dropna()
    returns = returns.loc[market_ret.index]

    mean_daily = returns.mean()
    ann_return = mean_daily * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)

    betas = {}
    for t in returns.columns:
        cov = np.cov(returns[t], market_ret)[0, 1]
        betas[t] = cov / np.var(market_ret)

    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)
    param_var_95 = mean_daily + norm.ppf(0.05) * returns.std()

    summary = pd.DataFrame({
        "Ann_Return": ann_return,
        "Ann_Volatility": ann_vol,
        "Beta": pd.Series(betas),
        "Hist_VaR_95": var_95,
        "Hist_VaR_99": var_99,
        "Param_VaR_95": param_var_95,
    })
    return returns, summary


if __name__ == "__main__":
    prices, market = load_data()
    returns, summary = compute_risk_metrics(prices, market)

    prices.to_csv("portfolio_prices.csv")
    market.to_csv("market_index.csv")
    returns.to_csv("daily_returns.csv")
    summary.to_csv("risk_summary.csv")

    print(summary.round(4))
