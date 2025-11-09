'''Real time stock predictor using XGBoost'''
#A basic desicion model that says the probability of the price of a particular stock going up or down using random forest regression
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import precision_score


def fetch_data(ticker, period='5y'):
    """Downloads historical data and fixes column structure."""
    print(f"Downloading data for {ticker}...")
    df = yf.download(tickers=ticker, period=period)
    
    # Fix MultiIndex issue if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    market_df = fetch_data('^NSEI')
    df['Nifty50_Close'] = market_df['Close']
        
    return df

def add_technical_features(df):
    #To calculate average over 50 days
    df['SMA_50']=df['Close'].rolling(window=50).mean()#Long term 50 day trend
    df['SMA_20']=df['Close'].rolling(window=20).mean()#short term 20 day trend
    df['Daily_Return']=df['Close'].pct_change()#percentage change from yesterday
    # Rolling Standard Deviation (how much the price swings around the average)
    df['Volatility'] = df['Close'].rolling(window=50).std()
    # --- RSI Calculation ---
    # 1. Calculate daily price changes
    delta = df['Close'].diff()

    # 2. Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # 3. Calculate standard 14-day rolling averages for gains and losses
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # 4. Calculate RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df=df.dropna()
    return df

def prepare_data(df):
    #shifting everything up to get tomorrow's close price
    df["Tomorrow_Close"]=df['Close'].shift(-1) 
    #Target= 1 if tmrw's close>close else 0
    df['Target']=(df['Tomorrow_Close']>df['Close']).astype(int)

    #print(df[['Close','Tomorrow_Close','Target']].tail())

    # 1. Define our predictors (features) and our target
    predictors = ['SMA_50', 'SMA_20', 'Daily_Return','Volatility','RSI']
    target = 'Target'

    #Training 
    # 2. Split the data based on time
    # Let's use everything before 2024 for training, and everything starting from 2024 for testing.
    train = df[df.index < '2024-01-01']
    test = df[df.index >= '2024-01-01']

    # 3. Create the training and testing sets
    X_train = train[predictors]
    y_train = train[target]
    X_test = test[predictors]
    y_test = test[target]
    return X_train, y_train, X_test, y_test

def train_and_predict(X_train, y_train, X_test):
    # n_estimators: Number of boosting rounds (trees)
    # learning_rate: How much each tree contributes (lower is slower but often more accurate)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=1)
    model.fit(X_train,y_train)
    # Instead of just a 0 or 1, get the probability (confidence) of it being a 1
    probs = model.predict_proba(X_test)[:, 1]
    #"If probability is greater than 0.6, predict 1, otherwise 0"
    custom_preds = (probs > 0.6).astype(int)
    return model,probs,custom_preds

def evaluate_and_plot(df, X_test, y_test, custom_preds, model, ticker_name):
    new_score = precision_score(y_test, custom_preds)
    print(f"New Precision Score: {new_score}")

    # 1. Grab the very last row of data (most recent trading day)
    last_day = X_test.iloc[-1:]

    # 2. Ask the model for its confidence score that the next day is "UP"
    next_day_prob = model.predict_proba(last_day)[:, 1][0]

    print(f"Probability of {ticker_name} going UP next trading day: {next_day_prob:.2f}")
    test_with_preds = X_test.copy()
    test_with_preds['Prediction'] = custom_preds

    # 2. Calculate the return for the NEXT day
    # We use .shift(-1) to bring tomorrow's return into today's row, just like we did for the Target
    test_with_preds['Next_Day_Return'] = df['Daily_Return'].shift(-1)
    test_with_preds['Strategy_Return'] = test_with_preds['Next_Day_Return'] * test_with_preds['Prediction']
    # Calculate the cumulative returns for both the stock itself (Buy & Hold) and your strategy
    test_with_preds['Buy_and_Hold'] = (1 + test_with_preds['Daily_Return']).cumprod()
    test_with_preds['Strategy_Equity'] = (1 + test_with_preds['Strategy_Return']).cumprod()

    # Plot them together to compare
    plt.figure(figsize=(12, 6))
    plt.plot(test_with_preds['Buy_and_Hold'], label=f'Buy and Hold ({ticker_name})', color='grey', alpha=0.5)
    plt.plot(test_with_preds['Strategy_Equity'], label='Your Strategy', color='green', linewidth=2)
    plt.title('Strategy Performance vs Buy & Hold (2024-2025)')
    plt.legend()
    plt.show()


print("1.Tata Motors \n2.Adani Power \n")
n=int(input('Enter'))
if n==1:
    s='TATAMOTORS.NS'
    c='Tata Motors'
elif n==2:
    s='ADANIPOWER.NS'
    c='Adani Power'

df=fetch_data(s)



print(market_df.head())

df=add_technical_features(df)

X_train,y_train,X_test,y_test=prepare_data(df)
    
model,probs,custom_preds=train_and_predict(X_train, y_train, X_test)

evaluate_and_plot(df, X_test, y_test, custom_preds, model,c)