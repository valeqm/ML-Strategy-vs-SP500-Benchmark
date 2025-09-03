"""
data_fetch.py
- Downloads a list of tickers (S&P 500) from Wikipedia
- Downloads OHLCV price data via yfinance
- Optional: can download macro data (FRED) or indicators (AlphaVantage)
Output: CSVs per ticker in ./data/raw/ and a combined prices parquet in ./data/processed/
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import argparse
import time
import requests

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

def fetch_sp500_tickers():
    """Fetch the list of S&P 500 tickers and sectors from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # raise error if request fails

    tables = pd.read_html(response.text)
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    sectors = df.set_index('Symbol')['GICS Sector'].to_dict()
    return tickers, sectors

def download_prices(tickers, start="2010-01-01", end=None, interval="1d"):
    """Download OHLCV prices for a list of tickers and save individual CSVs and a combined parquet."""
    import math
    end = end or datetime.today().strftime("%Y-%m-%d")
    all_dfs = []
    
    # Batch size (download 50 tickers per request)
    batch_size = 50
    n_batches = math.ceil(len(tickers) / batch_size)
    
    for i in range(n_batches):
        batch = tickers[i*batch_size:(i+1)*batch_size]
        df_batch = yf.download(batch, start=start, end=end, interval=interval, group_by='ticker', threads=True)
        
        for t in batch:
            if t in df_batch:
                df = df_batch[t].copy()
                df.index.name = "date"
                outpath = f"data/raw/{t}.parquet"
                df.to_parquet(outpath)
                all_dfs.append(df.assign(ticker=t))
        
        time.sleep(1)  # friendly pause to avoid throttling
    
    combined = pd.concat(all_dfs).reset_index()
    combined.to_parquet("data/processed/combined_prices.parquet", index=False)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--source", default="sp500", choices=["sp500", "custom"])
    parser.add_argument("--custom_file", default=None, help="CSV file with a 'ticker' column")
    args = parser.parse_args()

    if args.source == "sp500":
        tickers, sectors = fetch_sp500_tickers()
    else:
        df = pd.read_csv(args.custom_file)
        tickers = df['ticker'].tolist()
        sectors = {}

    print(f"Downloading {len(tickers)} tickers; start={args.start}")
    combined = download_prices(tickers[:200], start=args.start, end=args.end)  # default: first 200 tickers to avoid throttling
    print("Saved: data/processed/combined_prices.parquet")



