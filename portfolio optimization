"""
Week 3 — Mean-Variance Portfolio Optimization
Case study: 5-asset equity portfolio (AAPL, MSFT, JPM, XOM, JNJ)

Implements Markowitz mean-variance optimization (max-Sharpe and
minimum-variance) and traces the efficient frontier, using the daily
returns produced by risk_analysis.py.

Run:
    pip install pandas numpy scipy
    python portfolio_optimization.py
(expects daily_returns.csv produced by risk_analysis.py)
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

RISK_FREE_RATE = 0.02  # assumed annual risk-free rate
TRADING_DAYS = 252


def load_returns(path="daily_returns.csv"):
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def optimize(returns):
    tickers = returns.columns.tolist()
    n = len(tickers)

    mean_ret = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS

    def port_return(w):
        return np.dot(w, mean_ret)

    def port_vol(w):
        return np.sqrt(w.T @ cov_matrix @ w)

    def neg_sharpe(w):
        return -(port_return(w) - RISK_FREE_RATE) / port_vol(w)

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init = np.array([1 / n] * n)

    res_sharpe = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=constraints)
    res_minvar = minimize(port_vol, init, method="SLSQP", bounds=bounds, constraints=constraints)
    w_eq = init

    # Efficient frontier
    target_returns = np.linspace(mean_ret.min(), mean_ret.max(), 50)
    frontier_vol = []
    for target in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: port_return(w) - target},
        )
        res = minimize(port_vol, init, method="SLSQP", bounds=bounds, constraints=cons)
        frontier_vol.append(res.fun if res.success else np.nan)

    frontier = pd.DataFrame({"Return": target_returns, "Volatility": frontier_vol})

    def var95_1day(w):
        daily_ret = np.dot(returns, w)
        mu, sigma = daily_ret.mean(), daily_ret.std()
        return mu + norm.ppf(0.05) * sigma

    results = {
        "Max_Sharpe": {
            "weights": dict(zip(tickers, res_sharpe.x)),
            "return": port_return(res_sharpe.x),
            "vol": port_vol(res_sharpe.x),
            "sharpe": -res_sharpe.fun,
            "var95_1day": var95_1day(res_sharpe.x),
        },
        "Min_Variance": {
            "weights": dict(zip(tickers, res_minvar.x)),
            "return": port_return(res_minvar.x),
            "vol": port_vol(res_minvar.x),
            "sharpe": (port_return(res_minvar.x) - RISK_FREE_RATE) / port_vol(res_minvar.x),
            "var95_1day": var95_1day(res_minvar.x),
        },
        "Equal_Weight": {
            "weights": dict(zip(tickers, w_eq)),
            "return": port_return(w_eq),
            "vol": port_vol(w_eq),
            "sharpe": (port_return(w_eq) - RISK_FREE_RATE) / port_vol(w_eq),
            "var95_1day": var95_1day(w_eq),
        },
    }

    return results, frontier


if __name__ == "__main__":
    returns = load_returns()
    results, frontier = optimize(returns)

    frontier.to_csv("efficient_frontier.csv", index=False)
    with open("portfolio_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(json.dumps(results, indent=2, default=float))
