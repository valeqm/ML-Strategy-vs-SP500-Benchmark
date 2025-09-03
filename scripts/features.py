"""
features.py
- Generates technical and statistical features for each ticker
- Calculates returns, moving averages, volatility, momentum indicators, and price structure
- Adds time-based features and target variable for future returns
Output: a combined features parquet file in ./data/processed/features.parquet
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import skew, kurtosis
from joblib import Parallel, delayed
import os

os.makedirs("data/processed", exist_ok=True)

def prepare_base(df):
    """Prepare the base dataframe: parse dates, sort, and set index."""
    # expects columns: date, Open, High, Low, Close, Adj Close (optional), Volume, ticker
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker','date'])
    df = df.set_index('date')
    return df

def process_ticker(t, g):
    """Process a single ticker and return its dataframe with features."""
    gg = g.copy().sort_index()
    
    # ensure 'close' column
    if 'Adj Close' in gg.columns:
        gg['close'] = gg['Adj Close']
    else:
        gg['close'] = gg['Close']

    # basic returns
    gg['ret_1d'] = gg['close'].pct_change(1)
    gg['ret_5d'] = gg['close'].pct_change(5)
    gg['ret_10d'] = gg['close'].pct_change(10)

    # moving averages
    gg['ma_5'] = gg['close'].rolling(5).mean()
    gg['ma_10'] = gg['close'].rolling(10).mean()
    gg['ma_20'] = gg['close'].rolling(20).mean()
    gg['ema_20'] = gg['close'].ewm(span=20, adjust=False).mean()

    # volatility
    gg['rolling_std_10'] = gg['ret_1d'].rolling(10).std()
    gg['rolling_std_20'] = gg['ret_1d'].rolling(20).std()

    # momentum / indicators via pandas_ta
    try:
        rsi = ta.rsi(gg['close'], length=14)
        macd = ta.macd(gg['close'])  # DataFrame with MACD, MACDh, MACDs
        bbands = ta.bbands(gg['close'], length=20)
        stoch = ta.stoch(gg['High'], gg['Low'], gg['close'])
        atr = ta.atr(gg['High'], gg['Low'], gg['close'], length=14)
        obv = ta.obv(gg['close'], gg['Volume'])
    except Exception as e:
        print(f"pandas_ta error in {t}:", e)
        rsi = pd.Series(np.nan, index=gg.index)
        macd = pd.DataFrame(index=gg.index)
        bbands = pd.DataFrame(index=gg.index)
        stoch = pd.DataFrame(index=gg.index)
        atr = pd.Series(np.nan, index=gg.index)
        obv = pd.Series(np.nan, index=gg.index)

    gg['rsi_14'] = rsi
    # MACD
    if isinstance(macd, pd.DataFrame):
        for col in macd.columns:
            gg[col] = macd[col]
    if isinstance(bbands, pd.DataFrame):
        for col in bbands.columns:
            gg[col] = bbands[col]
    if isinstance(stoch, pd.DataFrame):
        for col in stoch.columns:
            gg[col] = stoch[col]
    gg['atr_14'] = atr
    gg['obv'] = obv.ffill()

    # price structure
    gg['52w_high'] = gg['close'].rolling(window=252, min_periods=1).max()
    gg['52w_low'] = gg['close'].rolling(window=252, min_periods=1).min()
    gg['dist_52w_high'] = (gg['52w_high'] - gg['close']) / gg['52w_high']
    gg['dist_52w_low'] = (gg['close'] - gg['52w_low']) / gg['52w_low']

    # statistical features
    gg['skew_20'] = gg['ret_1d'].rolling(20).apply(lambda x: skew(x.dropna()), raw=False)
    gg['kurt_20'] = gg['ret_1d'].rolling(20).apply(lambda x: kurtosis(x.dropna()), raw=False)
    gg['zscore_ret_20'] = (gg['ret_1d'] - gg['ret_1d'].rolling(20).mean()) / gg['ret_1d'].rolling(20).std()

    # time-based features
    gg['dayofweek'] = gg.index.dayofweek
    gg['is_weekend'] = gg['dayofweek'] >= 5
    gg['month'] = gg.index.month

    # target variable: future return next 5 days
    gg['future_5d_ret'] = gg['close'].pct_change(5).shift(-5)
    gg['target_5d_up'] = (gg['future_5d_ret'] > 0.02).astype(int)  # threshold adjustable

    gg['ticker'] = t
    return gg.reset_index()

def generate_features(df, n_jobs=-1):
    """Generate features for all tickers in parallel and save to parquet."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_ticker)(t, g) for t, g in df.groupby('ticker')
    )
    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(['ticker','date'])

    # create dummy variables for month
    month_dummies = pd.get_dummies(final['month'], prefix='m')
    final = pd.concat([final, month_dummies], axis=1)

    final.to_parquet("data/processed/features.parquet", index=False)
    print("Saved features to data/processed/features.parquet. Columns:", len(final.columns))
    return final

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/combined_prices.parquet")
    df = prepare_base(df)
    final = generate_features(df, n_jobs=-1)  # use all available cores



