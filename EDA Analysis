"""
Week 1 — Data Exploration and Preprocessing for Financial Analytics
Case study: AAPL daily OHLCV prices, Feb 2015 - Feb 2017

Source data: https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv
(a public, freely available republication of historical Yahoo Finance data)

Run:
    pip install pandas numpy matplotlib seaborn
    python eda_analysis.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

RAW_CSV = "raw_data.csv"  # download from the source URL above
OUT_DIR = "charts"


def load_and_clean(path=RAW_CSV):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(
        columns={
            "AAPL.Open": "Open",
            "AAPL.High": "High",
            "AAPL.Low": "Low",
            "AAPL.Close": "Close",
            "AAPL.Volume": "Volume",
            "AAPL.Adjusted": "Adj_Close",
        },
        inplace=True,
    )

    report = {}

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    report["date_range"] = (str(df["Date"].min()), str(df["Date"].max()))

    # Duplicates
    dupes = df.duplicated(subset="Date").sum()
    report["duplicate_dates"] = int(dupes)
    df = df.drop_duplicates(subset="Date")

    # Chronological order
    df = df.sort_values("Date").reset_index(drop=True)

    # Trading-day continuity check
    df["gap_days"] = df["Date"].diff().dt.days
    large_gaps = df[df["gap_days"] > 5]
    report["large_gaps"] = large_gaps[["Date", "gap_days"]].astype(str).to_dict("records")

    # Missing values
    report["missing_values"] = df.isna().sum().to_dict()

    # Logical consistency: High must be >= Low, Open, Close; Low must be <= Open, Close
    bad_rows = df[
        (df["High"] < df["Low"])
        | (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    ]
    report["logical_inconsistencies"] = len(bad_rows)

    # Outlier detection on daily returns (IQR method)
    df["Daily_Return"] = df["Close"].pct_change() * 100
    q1, q3 = df["Daily_Return"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df["Daily_Return"] < lower) | (df["Daily_Return"] > upper)]
    report["return_outliers_count"] = len(outliers)
    report["return_outlier_dates"] = outliers[["Date", "Daily_Return"]].round(2).astype(str).to_dict("records")

    # Volume outliers (z-score)
    df["Volume_Z"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    vol_outliers = df[df["Volume_Z"].abs() > 3]
    report["volume_outliers_count"] = len(vol_outliers)

    # Feature engineering
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility_20d"] = df["Daily_Return"].rolling(20).std()

    return df, report


def make_charts(df, out_dir=OUT_DIR):
    import os

    os.makedirs(out_dir, exist_ok=True)

    # 1. Price trend with moving averages
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["Date"], df["Close"], label="Close Price", color="#1f77b4", linewidth=1.2)
    ax.plot(df["Date"], df["MA20"], label="20-Day MA", color="orange", linewidth=1)
    ax.plot(df["Date"], df["MA50"], label="50-Day MA", color="green", linewidth=1)
    ax.set_title("AAPL Closing Price with Moving Averages (2015-2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/price_trend.png")
    plt.close()

    # 2. Daily return distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df["Daily_Return"].dropna(), bins=40, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Distribution of Daily Returns (%)")
    ax.set_xlabel("Daily Return (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/return_distribution.png")
    plt.close()

    # 3. Volume with outliers flagged
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["Date"], df["Volume"], width=1.5, color="#4c72b0", alpha=0.6)
    outliers = df[df["Volume_Z"].abs() > 3]
    ax.scatter(outliers["Date"], outliers["Volume"], color="red", zorder=5, label="Volume Outliers (|z|>3)")
    ax.set_title("Daily Trading Volume with Outliers Flagged")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/volume_outliers.png")
    plt.close()

    # 4. Correlation heatmap
    corr_cols = ["Open", "High", "Low", "Close", "Volume", "Daily_Return"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix of Key Variables")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/correlation_heatmap.png")
    plt.close()

    # 5. Return spread by direction label
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(data=df, x="direction", y="Daily_Return", ax=ax)
    ax.set_title("Daily Return Spread by Market Direction Label")
    ax.set_xlabel("Direction")
    ax.set_ylabel("Daily Return (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/return_by_direction.png")
    plt.close()


if __name__ == "__main__":
    df, report = load_and_clean()
    df.to_csv("cleaned_data.csv", index=False)
    with open("eda_report.json", "w") as f:
        json.dump(report, f, default=str, indent=2)
    make_charts(df)
    print(json.dumps(report, default=str, indent=2))
    print(df[["Close", "Daily_Return", "Volume"]].describe())
