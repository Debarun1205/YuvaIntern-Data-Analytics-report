"""
Week 3 — Chart generation for the risk analysis and portfolio optimization report.
Run after risk_analysis.py and portfolio_optimization.py have produced their CSV/JSON outputs.

Run:
    pip install pandas numpy matplotlib seaborn
    python generate_charts.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

OUT_DIR = "charts"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    prices = pd.read_csv("portfolio_prices.csv", index_col="Date", parse_dates=True)
    returns = pd.read_csv("daily_returns.csv", index_col="Date", parse_dates=True)
    summary = pd.read_csv("risk_summary.csv", index_col=0)
    frontier = pd.read_csv("efficient_frontier.csv")
    with open("portfolio_results.json") as f:
        results = json.load(f)

    tickers = prices.columns.tolist()
    colors = sns.color_palette("tab10", len(tickers))

    # 1. Normalized price performance
    fig, ax = plt.subplots(figsize=(9, 4.5))
    norm_prices = prices / prices.iloc[0] * 100
    for i, t in enumerate(tickers):
        ax.plot(norm_prices.index, norm_prices[t], label=t, color=colors[i])
    ax.set_title("Normalized Price Performance (2013=100)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_performance.png")
    plt.close()

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(returns.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix of Daily Returns")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png")
    plt.close()

    # 3. Risk-return scatter
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, t in enumerate(tickers):
        ax.scatter(summary.loc[t, "Ann_Volatility"], summary.loc[t, "Ann_Return"], s=120, color=colors[i], label=t)
        ax.annotate(f"{t} (\u03b2={summary.loc[t,'Beta']:.2f})",
                    (summary.loc[t, "Ann_Volatility"], summary.loc[t, "Ann_Return"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk-Return Profile by Asset (labeled with Beta)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/risk_return_scatter.png")
    plt.close()

    # 4. VaR comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(tickers))
    w = 0.35
    ax.bar(x - w / 2, summary["Hist_VaR_95"] * 100, w, label="95% VaR", color="#d62728")
    ax.bar(x + w / 2, summary["Hist_VaR_99"] * 100, w, label="99% VaR", color="#8c1515")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Daily VaR (%)")
    ax.set_title("Historical Value-at-Risk by Asset (1-Day Horizon)")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/var_comparison.png")
    plt.close()

    # 5. Efficient frontier
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(frontier["Volatility"], frontier["Return"], color="#1f77b4", label="Efficient Frontier", linewidth=2)
    for i, t in enumerate(tickers):
        ax.scatter(summary.loc[t, "Ann_Volatility"], summary.loc[t, "Ann_Return"], s=100, color=colors[i], label=t, zorder=5)
    ax.scatter(results["Max_Sharpe"]["vol"], results["Max_Sharpe"]["return"], marker="*", s=400,
               color="gold", edgecolor="black", label="Max Sharpe Portfolio", zorder=6)
    ax.scatter(results["Min_Variance"]["vol"], results["Min_Variance"]["return"], marker="D", s=150,
               color="green", edgecolor="black", label="Min Variance Portfolio", zorder=6)
    ax.scatter(results["Equal_Weight"]["vol"], results["Equal_Weight"]["return"], marker="s", s=120,
               color="gray", edgecolor="black", label="Equal-Weight Portfolio", zorder=6)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier and Optimized Portfolios")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/efficient_frontier.png")
    plt.close()

    # 6. Weight comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w_sharpe = [results["Max_Sharpe"]["weights"][t] for t in tickers]
    w_minvar = [results["Min_Variance"]["weights"][t] for t in tickers]
    w_eq = [results["Equal_Weight"]["weights"][t] for t in tickers]
    x = np.arange(len(tickers))
    width = 0.25
    ax.bar(x - width, w_eq, width, label="Equal Weight")
    ax.bar(x, w_minvar, width, label="Min Variance")
    ax.bar(x + width, w_sharpe, width, label="Max Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Allocation Comparison Across Strategies")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/weight_comparison.png")
    plt.close()

    print("All charts saved to", OUT_DIR)


if __name__ == "__main__":
    main()
