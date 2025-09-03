# ML Strategy vs S&P 500 Benchmark

An end-to-end machine learning trading strategy project where we build, backtest, and evaluate ML-driven trading signals against the S&P 500 index (SPY) as a benchmark.

This is my submission for [Stock Markets Analytics Zoomcamp 2025](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp) final project, fulfilling the requirements of designing, implementing, and automating an algo-trading pipeline.

## Project Overview

The goal is to predict short-term stock returns using ML models and evaluate the profitability of a trading strategy compared to the S&P 500 benchmark.

The pipeline covers:

- **Data collection** – Download historical stock data (SP500) using `yfinance`.  
- **Feature engineering** – Generate >20 features (returns, rolling statistics, volatility, momentum, technical indicators).  
- **Modeling** – Train multiple ML models (Decision Tree, Random Forest, XGBoost).  
- **Trading strategy** – Define strategies based on model probabilities (threshold, probability-weighted).  
- **Backtest** – Simulate trades with reinvestment and calculate CAGR, Sharpe ratio, Max Drawdown.  
- **Benchmarking** – Compare strategy performance against SPY cumulative returns.  
- **Automation** – Full pipeline executed via `run_pipeline.py`.

## 📂 Repository Structure

```sh
ml-strategy-vs-sp500-benchmark/
│── data/                  # (not uploaded - fetched via scripts)
│── models/                # trained models (.joblib)
│── output/                # backtest results, metrics, dashboard data
│── scripts/
│   ├── data_fetch.py      # Download stock data
│   ├── features.py        # Feature engineering
│   ├── train_models.py    # Train ML models
│   ├── backtest.py        # Trading strategy + backtest
│   ├── benchmark.py       # Benchmark vs SPY
│── run_pipeline.py        # Orchestration of full pipeline
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

For more details on the script, [click here](/scripts/scripts.md).

## ⚙️ Installation

Clone the repo:

```sh
git clone https://github.com/yourname/ml-strategy-vs-sp500-benchmark.git
cd ml-strategy-vs-sp500-benchmark
```

Install dependencies:

```sh
pip install -r requirements.txt
```

## ▶️ Usage

Run the full pipeline (data → features → models → backtest → benchmark → output):

```sh
python run_pipeline.py --start 2010-01-01 --benchmark_symbol SPY
```

Options:

`--train` → force retraining models

`--end YYYY-MM-DD` → set end date

`--tickers_source sp500` → choose stock universe (SP500 by default)

Outputs:

`output/portfolio_threshold_series.csv` → backtest curve + metrics

`output/SPY_benchmark.csv` → benchmark curve

`output/dashboard_data.csv` → combined curves

`output/dashboard_metrics.csv` → strategy vs benchmark metrics

You can also download see the final outputs from my Google Drive :

🔗 [ML Strategy Results Drive Folder](https://drive.google.com/drive/folders/1KaguxuFx44iSP58NgmiXlZ422X80aZhg?usp=sharing) 

## 📊 Results

Example comparison chart:

**Strategy vs S&P 500 (SPY) Performance**

- Strategy CAGR: higher than benchmark
- Sharpe ratio: superior risk-adjusted returns
- Max Drawdown: analyzed for robustness

The pipeline shows how ML-driven signals can outperform a passive benchmark in certain configurations.

## 📈 Dashboard  

This dashboard compares the cumulative performance of the trading strategy against the SPY benchmark over time and metrics such as CAGR, Sharpe Ratio, and Max Drawdown.

🔗 [View in Looker Studio](https://lookerstudio.google.com/reporting/66ad0311-bd18-4965-9df0-046dc1f9228d)  

## Future Improvements

- Add stop-loss / risk management rules.
- Explore LSTM/Deep Learning models.
- Integrate with broker API for live trading.

## Author  

**Valeria Q.M** 

[![LinkedIn](https://img.shields.io/badge/-💼%20LinkedIn-white?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/valeriaqm/)
[![Credly](https://img.shields.io/badge/-Credly-white?style=flat&logo=credly&logoColor=FFA500)](https://www.credly.com/users/valeria-quijada)
[![Google Cloud Skill Boost](https://img.shields.io/badge/-Google%20Cloud%20Skills-white?style=flat&logo=googlecloud&logoColor=4285F4)](https://www.cloudskillsboost.google/public_profiles/36f6887c-3fbb-4cab-9f3b-74f534cf89b0?locale=es)
[![GitHub](https://img.shields.io/badge/-GitHub-white?style=flat&logo=github&logoColor=181717)](https://github.com/valeqm)
[![Reddit](https://img.shields.io/badge/-Reddit-white?style=flat&logo=reddit&logoColor=FF4500)](https://www.reddit.com/)


