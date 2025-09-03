# Algorithmic Trading Project - README

## data_fetch.py
**Purpose:** Download stock tickers and historical price data for the S&P 500.  

**Workflow:**  
- Retrieves the list of S&P 500 tickers from Wikipedia, including sector information.  
- Downloads OHLCV price data for each ticker via yfinance.  
- Handles batching of requests to avoid rate limits and allows incremental updates. 

**Features & Outputs:**  
- Individual Parquet files per ticker saved in `data/raw/`.  
- Combined dataset saved as `data/processed/combined_prices.parquet`.  
- Supports optional custom tickers CSV.  

**Notes:**  
- Data spans from a specified start date (default: 2010-01-01) to the present.  
- Used as input for feature generation and model training.

## features.py
**Purpose:** Generate technical, statistical, and time-based features from the combined price dataset.  

**Workflow:**  
- Processes each ticker individually.  
- Calculates basic returns (1d, 5d, 10d).  
- Computes moving averages (5, 10, 20 days) and EMA(20).  
- Measures volatility (rolling 10 and 20-day std).  
- Generates momentum indicators via pandas_ta: RSI(14), MACD, Bollinger Bands, Stochastic Oscillator, ATR(14), OBV.  
- Determines price structure features: 52-week high/low, distance to high/low.  
- Computes rolling statistics: skew, kurtosis, z-score of returns.  
- Adds time-based features: day of week, weekend flag, month (with dummy variables).  
- Creates target variable: 5-day forward return (`future_5d_ret`) and binary target (`target_5d_up`).  

**Features & Outputs:**  
- Over 20 features including numeric and dummy variables.  
- Saves final dataset to `data/processed/features.parquet`.  

**Notes:**  
- Ensures clean, sorted data indexed by date per ticker.  
- Ready for model training.

## train_models.py
**Purpose:** Train predictive models for 5-day upward price movement.  

**Workflow:**  
- Loads `features.parquet` and prepares X (features) and y (`target_5d_up`).  
- Splits data by date into train, validation, and test sets to prevent leakage.  
- Trains multiple models: Decision Tree, Random Forest, XGBoost.  
- Performs hyperparameter tuning using `RandomizedSearchCV`.  
- Handles class imbalance via class weights and XGBoost `scale_pos_weight`.  

**Evaluation:**  
- Computes validation metrics: AUC and Average Precision.  
- Selects best model based on validation precision.  

**Outputs:**  
- Saves best model as `models/best_model.joblib`.  
- Saves scaler (`models/scaler.joblib`) and selected features (`models/feature_cols.joblib`).  

**Notes:**  
- All models trained on numeric + dummy features.  
- Models can be used directly in backtesting or live prediction pipelines.

## backtest.py
**Purpose:** Simulate trading strategies based on model predictions. 

**Workflow:**  
- Loads trained model and features dataset.  
- Generates predicted probabilities of price increase.  
- Applies trading strategies:  
  1. **Threshold-based:** enter position if probability > threshold.  
  2. **Probability-weighted:** allocate capital proportionally to predicted probability.  
- Calculates daily P&L per ticker.  
- Computes portfolio-level metrics: cumulative portfolio value, drawdowns, CAGR, annualized volatility, Sharpe ratio.  

**Outputs:**  
- Detailed portfolio CSV with daily returns (`output/portfolio_threshold_series.csv`).  
- Predictions dataset (`output/preds_with_data.parquet`).  
- Metrics summary CSV (`output/backtest_metrics.csv`).  

**Notes:**  
- Supports iterative reinvestment and multiple strategies.  
- Ready for benchmark comparison.

## benchmark.py
**Purpose:** Download benchmark data and compare strategy performance.  

**Workflow:**  
- Downloads benchmark (default SPY) via yfinance.  
- Calculates daily returns, cumulative performance, and drawdowns.  
- Aligns benchmark dates with strategy portfolio data.  
- Computes metrics for both strategy and benchmark: CAGR, Sharpe ratio, Max Drawdown.  

**Outputs:**  
- Benchmark curve CSV (`output/SPY_benchmark.csv`).  
- Comparative metrics CSV (`output/benchmark_comparison.csv`).  
- Console printout of strategy vs. benchmark performance.  

**Notes:**  
- Enables assessment of strategy profitability relative to a realistic market benchmark.

## run_pipeline.py
**Purpose:** Execute the full end-to-end workflow for data download, feature generation, model training, backtesting, and benchmark comparison. 

**Workflow:**  
1. Downloads ticker data using `data_fetch.py`.  
2. Generates features using `features.py`.  
3. Trains models using `train_models.py` if models do not exist or if training is forced.  
4. Runs backtesting via `backtest.py`.  
5. Executes `benchmark.py` to download benchmark and compute comparative metrics.  
6. Loads backtest and benchmark results and normalizes cumulative strategy curves.  
7. Combines portfolio and benchmark curves for visualization or dashboard use.  
8. Saves combined data (`output/dashboard_data.csv`) and combined metrics (`output/dashboard_metrics.csv`).  

**Outputs:**  
- Normalized strategy and benchmark cumulative curves.  
- Dashboard-ready combined CSV files.  
- Console outputs guiding the workflow execution. 

**Notes:**  
- Modular and robust; allows rerunning the pipeline end-to-end.  
- Supports arguments for start/end dates, ticker sources, and benchmark symbols.  
- Ensures reproducibility and incremental updates of models and data.