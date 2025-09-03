"""
benchmark.py
- Downloads a benchmark (e.g., SPY) and calculates returns and cumulative curve.
- Compares a backtested strategy portfolio with the benchmark.
- Calculates key metrics: CAGR, annual volatility, Sharpe ratio, max drawdown.
- Saves benchmark curve and comparison metrics to the output folder.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_benchmark(start="2010-01-01", end=None, symbol="SPY"):
    """Download a benchmark (e.g., SPY) and calculate returns and cumulative curve."""
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)

    # Flatten columns in case of MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    if "Close" not in data.columns:
        raise ValueError(f"'Close' not found in downloaded data for {symbol}")

    data["Return"] = data["Close"].pct_change().fillna(0)
    data["Cumulative"] = (1 + data["Return"]).cumprod()
    return data

def compute_metrics(daily_ret):
    """Calculate key performance metrics for any return series."""
    dr = daily_ret.dropna()
    if len(dr) == 0:
        return {}

    total_return = (1 + dr).prod() - 1
    n_days = dr.shape[0]
    years = n_days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = (dr.mean() * 252) / (ann_vol + 1e-9)

    cum = (1 + dr).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd
    }

def compare_with_strategy(portfolio_csv="output/portfolio_threshold_series.csv",
                          start="2010-01-01", end=None, symbol="SPY"):
    """Compare a backtested strategy portfolio with a benchmark."""

    # Load backtest portfolio
    port = pd.read_csv(portfolio_csv, parse_dates=["Date"], index_col="Date")

    # Daily returns of the strategy
    port["DailyReturn"] = port["Portfolio"].pct_change().fillna(0)

    # Download benchmark
    spy = download_benchmark(start=port.index.min(), end=port.index.max(), symbol=symbol)

    # Align dates
    df = pd.DataFrame({
        "Strategy": port["Strategy_Cumulative"],  # already normalized in backtest
        symbol: spy["Cumulative"] / spy["Cumulative"].iloc[0]
    }).dropna()

    # Save benchmark cumulative curve
    os.makedirs("output", exist_ok=True)
    benchmark_csv = f"output/{symbol}_benchmark.csv"
    df[[symbol]].to_csv(benchmark_csv)
    print(f"Benchmark saved to {benchmark_csv}")

    # Calculate metrics
    metrics_strategy = compute_metrics(port["DailyReturn"])
    metrics_bench = compute_metrics(spy["Return"].loc[df.index])

    # Save comparative metrics
    pd.DataFrame([{
        "Metric": "CAGR",
        "Strategy": metrics_strategy["CAGR"],
        symbol: metrics_bench["CAGR"]
    }, {
        "Metric": "AnnVol",
        "Strategy": metrics_strategy["AnnVol"],
        symbol: metrics_bench["AnnVol"]
    }, {
        "Metric": "Sharpe",
        "Strategy": metrics_strategy["Sharpe"],
        symbol: metrics_bench["Sharpe"]
    }, {
        "Metric": "MaxDD",
        "Strategy": metrics_strategy["MaxDD"],
        symbol: metrics_bench["MaxDD"]
    }]).to_csv("output/benchmark_comparison.csv", index=False)

    print("Comparative metrics saved in 'output/benchmark_comparison.csv'")

    # Print to console
    print("=== Benchmark Comparison ===")
    print(f"Strategy CAGR: {metrics_strategy['CAGR']:.2%}")
    print(f"{symbol} CAGR: {metrics_bench['CAGR']:.2%}")
    print(f"Strategy Sharpe: {metrics_strategy['Sharpe']:.2f}")
    print(f"{symbol} Sharpe: {metrics_bench['Sharpe']:.2f}")
    print(f"Strategy MaxDD: {metrics_strategy['MaxDD']:.2%}")
    print(f"{symbol} MaxDD: {metrics_bench['MaxDD']:.2%}")

    return df

if __name__ == "__main__":
    compare_with_strategy(
        portfolio_csv="output/portfolio_threshold_series.csv",
        symbol="SPY"
    )
