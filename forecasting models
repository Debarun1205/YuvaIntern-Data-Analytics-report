"""
Week 2 — Financial Forecasting with Time Series Analysis
Case study: AAPL daily closing price, Feb 2015 - Feb 2017

Compares ARIMA(1,1,1) against Holt's damped-trend Exponential Smoothing,
with stationarity testing, ACF/PACF diagnostics, and residual analysis.

Run:
    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
    python forecasting_models.py
(expects cleaned_data.csv produced by week1_data_exploration/eda_analysis.py)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

N_TEST = 30  # trading days held out for evaluation


def load_series(path="cleaned_data.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").asfreq("B")
    df["Close"] = df["Close"].interpolate()
    return df["Close"]


def run_diagnostics(train, out_dir="charts"):
    import os

    os.makedirs(out_dir, exist_ok=True)

    adf_level = adfuller(train.dropna())
    diff1 = train.diff().dropna()
    adf_diff = adfuller(diff1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(diff1, ax=axes[0], lags=30, title="ACF of Differenced Close Price")
    plot_pacf(diff1, ax=axes[1], lags=30, title="PACF of Differenced Close Price")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/acf_pacf.png")
    plt.close()

    decomp = seasonal_decompose(train, model="additive", period=5, extrapolate_trend="freq")
    fig = decomp.plot()
    fig.set_size_inches(9, 7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/decomposition.png")
    plt.close()

    return {
        "adf_level_stat": adf_level[0],
        "adf_level_p": adf_level[1],
        "adf_diff_stat": adf_diff[0],
        "adf_diff_p": adf_diff[1],
    }


def fit_and_forecast(train, test, out_dir="charts"):
    # ARIMA(1,1,1)
    fit_arima = ARIMA(train, order=(1, 1, 1)).fit()
    fc = fit_arima.get_forecast(steps=N_TEST)
    pred_arima = fc.predicted_mean
    ci_arima = fc.conf_int(alpha=0.05)

    mae_arima = mean_absolute_error(test, pred_arima)
    rmse_arima = np.sqrt(mean_squared_error(test, pred_arima))
    mape_arima = mean_absolute_percentage_error(test, pred_arima) * 100

    # Holt's damped-trend exponential smoothing
    fit_es = ExponentialSmoothing(train, trend="add", seasonal=None, damped_trend=True).fit()
    pred_es = fit_es.forecast(N_TEST)

    mae_es = mean_absolute_error(test, pred_es)
    rmse_es = np.sqrt(mean_squared_error(test, pred_es))
    mape_es = mean_absolute_percentage_error(test, pred_es) * 100

    # Forecast plots
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train.index[-60:], train.iloc[-60:], label="Train (last 60d)", color="#1f77b4")
    ax.plot(test.index, test, label="Actual", color="black", linewidth=1.5)
    ax.plot(test.index, pred_arima, label="ARIMA(1,1,1) Forecast", color="#d62728", linestyle="--")
    ax.fill_between(test.index, ci_arima.iloc[:, 0], ci_arima.iloc[:, 1], color="#d62728", alpha=0.15, label="95% CI")
    ax.set_title("ARIMA(1,1,1) Forecast vs. Actual (Last 30 Trading Days)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/arima_forecast.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train.index[-60:], train.iloc[-60:], label="Train (last 60d)", color="#1f77b4")
    ax.plot(test.index, test, label="Actual", color="black", linewidth=1.5)
    ax.plot(test.index, pred_es, label="Holt's ES Forecast", color="#2ca02c", linestyle="--")
    ax.set_title("Holt's Damped Trend Exponential Smoothing Forecast vs. Actual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/holt_forecast.png")
    plt.close()

    # Residual diagnostics for ARIMA
    residuals = fit_arima.resid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title("ARIMA Residuals over Time")
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Residual Distribution")
    plot_acf(residuals, ax=axes[1, 0], lags=30, title="ACF of Residuals")
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/residual_diagnostics.png")
    plt.close()

    return {
        "arima": {"order": "(1,1,1)", "mae": mae_arima, "rmse": rmse_arima, "mape": mape_arima,
                  "aic": fit_arima.aic, "bic": fit_arima.bic},
        "holt": {"mae": mae_es, "rmse": rmse_es, "mape": mape_es},
    }


if __name__ == "__main__":
    ts = load_series()
    train, test = ts.iloc[:-N_TEST], ts.iloc[-N_TEST:]

    stationarity = run_diagnostics(train)
    print("Stationarity tests:", json.dumps(stationarity, indent=2))

    metrics = fit_and_forecast(train, test)
    print("Model metrics:", json.dumps(metrics, indent=2, default=float))

    with open("model_metrics.json", "w") as f:
        json.dump({**stationarity, **metrics}, f, indent=2, default=float)
