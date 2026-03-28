# 📈 XGBoost Algo Trading System 
A professional-grade machine learning pipeline and algorithmic trading backtester designed for the Indian Stock Market (NSE). This project leverages an **XGBoost Classifier** to predict daily price movements and evaluates trading strategies using strictly stationary technical indicators, institutional-grade risk metrics, and realistic market friction.

## 🚀 Key Upgrades & Quantitative Features

This is not a standard machine learning tutorial. This engine has been engineered to avoid classic quantitative finance pitfalls (like look-ahead bias, the "flatline precision trap", and premature early stopping):

* **Advanced Stationary Feature Engineering:** Converts raw prices into highly predictive, stationary data points:
  * **Garman-Klass Realized Volatility:** A highly efficient OHLC-based volatility estimator.
  * **Volatility Regime Ratio:** Short/long vol ratio acting as a volatility clustering proxy.
  * **VWAP Distance:** Volume-weighted mean-reversion signals.
  * **Rolling Return Autocorrelation:** Detects momentum vs. mean-reversion market regimes.
* **Institutional Risk Metrics:** Goes beyond simple "Accuracy" to calculate Hedge Fund standard metrics: **Sharpe, Sortino, Calmar, Max Drawdown, Win Rate (Active Days), and Profit Factor.**
* **Dual-Mode Visualizations:** Generates ultra-fast static `Matplotlib` dashboards AND highly interactive `Plotly` HTML Tear Sheets for deep visual analysis.
* **Realistic Backtesting Engine:** Incorporates a `0.15%` round-trip friction cost (covering Delivery STT, Brokerage, and Exchange fees) and strictly drops incomplete "live" NSE bars during trading hours to prevent data leakage.
* **Optimized ML Pipeline:** Uses `HalvingRandomSearchCV` with `TimeSeriesSplit` across 5 market regimes. Optimizes for **PR-AUC (Average Precision)** rather than raw precision to prevent the algorithm from "flatlining" and missing major market breakouts.

## 📂 Project Structure

| File | Description | 
| ----- | ----- | 
| `model_engine.py` | The "Brain". Handles data pipelines, timezone-aware Yahoo Finance fetching, local CSV caching, stationary feature engineering, and the XGBoost Time-Series cross-validation pipeline. | 
| `backtest_engine.py` | The "Execution". Imports the brain, simulates trades, deducts transaction costs, calculates institutional performance metrics, and generates the visual dashboards. | 
| `requirements.txt` | [cite_start]Python dependencies required to run the pipeline. | 

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/rohithkrishnaak/realtime_stock_predictor.git](https://github.com/rohithkrishnaak/realtime_stock_predictor.git)
   cd realtime_stock_predictor
