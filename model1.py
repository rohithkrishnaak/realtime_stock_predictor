"""Real-time Indian Equity Predictor & Backtester using XGBoost"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, accuracy_score

def fetch_data(ticker, period='5y'):
    print(f"Downloading data for {ticker} & Nifty 50...")
    # Fetch target stock
    df = yf.download(tickers=ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Fetch Nifty 50 for broader market context
    market_df = yf.download(tickers='^NSEI', period=period, progress=False)
    if isinstance(market_df.columns, pd.MultiIndex):
        market_df.columns = market_df.columns.get_level_values(0)
        
    df['Nifty50_Close'] = market_df['Close']
    return df

def add_technical_features(df):
    """Engineers standard technical indicators and market context."""
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Returns and Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Nifty_Return'] = df['Nifty50_Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=50).std()
    
    # Bollinger Bands
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    return df

def prepare_data(df):
    """Prepares targets and dynamically splits data chronologically."""
    # Target: 1 if tomorrow's close is strictly greater than today's
    df["Tomorrow_Close"] = df['Close'].shift(-1) 
    df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
    
    # Drop the last row because we don't have tomorrow's data for it yet
    df = df.dropna(subset=['Tomorrow_Close'])

    # All engineered features
    predictors = [
        'SMA_50', 'SMA_20', 'Daily_Return', 'Nifty_Return', 
        'Volatility', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower'
    ]
    
    # Dynamic 80/20 Chronological Split
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train, y_train = train[predictors], train['Target']
    X_test, y_test = test[predictors], test['Target']
    
    return X_train, y_train, X_test, y_test, predictors, test.index

def train_and_predict(X_train, y_train, X_test):
    # Tuned XGBoost
    model = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    custom_preds = (probs > 0.6).astype(int) # High confidence threshold
    return model, custom_preds

def evaluate_and_plot(df, X_test, y_test, custom_preds, model, predictors, test_dates, ticker_name):
    # --- METRICS ---
    precision = precision_score(y_test, custom_preds, zero_division=0)
    accuracy = accuracy_score(y_test, custom_preds)
    
    # Strategy Returns calculation
    test_with_preds = X_test.copy()
    test_with_preds['Prediction'] = custom_preds
    test_with_preds['Close'] = df.loc[test_dates, 'Close']
    test_with_preds['Next_Day_Return'] = df.loc[test_dates, 'Daily_Return'].shift(-1)
    test_with_preds['Strategy_Return'] = test_with_preds['Next_Day_Return'] * test_with_preds['Prediction']
    
    # Cumulative math
    test_with_preds['Buy_and_Hold'] = (1 + test_with_preds['Next_Day_Return']).cumprod()
    test_with_preds['Strategy_Equity'] = (1 + test_with_preds['Strategy_Return']).cumprod()

    # Get latest prediction
    latest_data = df.iloc[-1:][predictors]
    next_day_prob = model.predict_proba(latest_data)[:, 1][0]
    
    print("\n" + "="*40)
    print(f"📊 RESULTS FOR {ticker_name.upper()}")
    print("="*40)
    print(f"Probability of going UP next trading day : {next_day_prob*100:.2f}%")
    print(f"Model Precision (Accuracy of 'Buy' calls): {precision*100:.2f}%")
    print(f"Overall Accuracy                       : {accuracy*100:.2f}%")
    print("-" * 40)
    print(f"Total Buy & Hold Return                  : {(test_with_preds['Buy_and_Hold'].iloc[-2] - 1)*100:.2f}%")
    print(f"Total Strategy Return                    : {(test_with_preds['Strategy_Equity'].iloc[-2] - 1)*100:.2f}%")
    print("="*40 + "\n")

    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), gridspec_kw={'height_ratios': [2, 1.5, 1]})
    
    # Panel 1: Price & Buy Signals
    ax1.plot(test_dates, test_with_preds['Close'], label='Close Price', color='black', alpha=0.7)
    ax1.plot(test_dates, test_with_preds['SMA_50'], label='50 SMA', color='orange', alpha=0.6)
    
    # Extract dates where model predicted '1'
    buy_signals = test_with_preds[test_with_preds['Prediction'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal (>60% Prob)', s=100, zorder=5)
    
    ax1.set_title(f'{ticker_name} - Price Action & Model Buy Signals')
    ax1.set_ylabel('Price (INR)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative Returns
    ax2.plot(test_dates, test_with_preds['Buy_and_Hold'], label='Buy & Hold', color='grey', alpha=0.6)
    ax2.plot(test_dates, test_with_preds['Strategy_Equity'], label='XGBoost Strategy', color='green', linewidth=2)
    ax2.set_title('Strategy Performance vs. Buy & Hold')
    ax2.set_ylabel('Cumulative Return Multiplier')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax3.barh(range(len(indices)), importances[indices], color='steelblue')
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels([predictors[i] for i in indices])
    ax3.set_title('XGBoost Feature Importances (What the model cares about)')
    ax3.set_xlabel('Relative Importance')

    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION ---
print("Select an equity to analyze:")
stock_map = {
    1: ('TATAMOTORS.NS', 'Tata Motors'),
    2: ('ADANIPOWER.NS', 'Adani Power'),
    3: ('ASHOKLEY.NS', 'Ashok Leyland'),
    4: ('COALINDIA.NS', 'Coal India'),
    5: ('MOTHERSON.NS', 'Motherson Sumi'),
    6: ('TATAPOWER.NS', 'Tata Power')
}

for key, val in stock_map.items():
    print(f"{key}. {val[1]}")

try:
    n = int(input('\nEnter choice (1-6): '))
    ticker_symbol, company_name = stock_map[n]
except (ValueError, KeyError):
    print("Invalid choice. Defaulting to Tata Motors.")
    ticker_symbol, company_name = stock_map[1]

# Pipeline
df = fetch_data(ticker_symbol)
df = add_technical_features(df)
X_train, y_train, X_test, y_test, predictors, test_dates = prepare_data(df)
model, custom_preds = train_and_predict(X_train, y_train, X_test)
evaluate_and_plot(df, X_test, y_test, custom_preds, model, predictors, test_dates, company_name)
