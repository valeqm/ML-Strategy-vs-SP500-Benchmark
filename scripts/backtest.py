"""
backtest.py
- Loads models and features.parquet
- Generates predictions (probabilities)
- Strategies:
   1) Threshold: go long if prob > threshold
   2) Prob-weighted: position size proportional to probability
   3) Top-N: each day, go long in the N tickers with highest probability
- Includes transaction costs and iterative simulation with reinvestment
- Calculates metrics: CAGR, annual volatility, Sharpe ratio, max drawdown
"""

import pandas as pd
import numpy as np
import joblib
import os

def load_models():
    """Load all trained models, scaler, and feature columns."""
    dt = joblib.load("models/decision_tree.joblib")
    rf = joblib.load("models/random_forest.joblib")
    xgb = joblib.load("models/xgboost.joblib")
    best = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/feature_cols.joblib")
    return dt, rf, xgb, best, scaler, features

def predict_and_backtest(df, model, scaler, features,
                         strategy="threshold", threshold=0.7,
                         initial_capital=100000):

    data = df.copy()
    X = data[features].fillna(0)
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]
    data['prob'] = probs

    # Strategy: determine positions
    if strategy == "threshold":
        data['position'] = (data['prob'] >= threshold).astype(int)
    elif strategy == "prob_weighted":
        data['position'] = data['prob']  # proportional weight
    else:
        raise ValueError("Unsupported strategy")

    # Daily returns per ticker
    data['next_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
    data['pnl'] = data['position'] * data['next_return']

    # Average daily return (equally weighted among selected tickers)
    daily_ret = data.groupby('date')['pnl'].mean().fillna(0)

    # Portfolio curve (in dollars)
    port = (1 + daily_ret).cumprod() * initial_capital

    # Compute overall metrics
    metrics = compute_metrics(daily_ret)

    # Detailed dataframe with portfolio curve
    port_df = pd.DataFrame({
        "Date": daily_ret.index,
        "DailyReturn": daily_ret.values,
        "Portfolio": port.values
    })

    # Add normalized cumulative strategy (starts at 1)
    port_df["Strategy_Cumulative"] = port_df["Portfolio"] / port_df["Portfolio"].iloc[0]

    # Drawdown
    roll_max = port_df["Portfolio"].cummax()
    port_df["Drawdown"] = (port_df["Portfolio"] - roll_max) / roll_max

    # Cumulative CAGR to each date
    port_df["CAGR_toDate"] = (
        (port_df["Portfolio"] / initial_capital) ** (252 / (np.arange(len(port_df)) + 1)) - 1
    )

    return port_df.set_index("Date"), metrics, data

def compute_metrics(daily_ret_series):
    """Calculate key portfolio metrics."""
    dr = daily_ret_series.dropna()
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

if __name__ == "__main__":
    # 1. Load features
    df = pd.read_parquet("data/processed/features.parquet")
    df = df.sort_values(['date', 'ticker'])

    # 2. Load models
    dt, rf, xgb, best, scaler, features = load_models()

    # 3. Backtest example with "best" model and threshold strategy
    port_df, metrics, data_with_preds = predict_and_backtest(
        df, best, scaler, features,
        strategy="threshold", threshold=0.2
    )

    # 4. Print metrics
    print("=== Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 5. Save results
    os.makedirs("output", exist_ok=True)
    port_df.to_csv("output/portfolio_threshold_series.csv")   # now with Strategy_Cumulative
    data_with_preds.to_parquet("output/preds_with_data.parquet")
    pd.DataFrame([metrics]).to_csv("output/backtest_metrics.csv", index=False)

    print("Results saved in 'output/'")


