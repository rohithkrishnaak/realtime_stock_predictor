#A basic desicion model that says the probability of the price of a particular stock going up or down using random forest regression
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# 1. Initialize (Create the model with some basic settings)
# n_estimators=100 means it will create 100 small decision trees.
# min_samples_split=100 helps prevent it from memorizing too much noise (overfitting).
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

def choose(s):
    global df
    df=yf.download(tickers=s,period='5y')
    # If the data has double headers (Ticker and Price), this flattens it to just Price
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

print("1.Tata Motors \n2.Adani Power \n")
n=int(input('Enter'))
if n==1:
    s='TATAMOTORS.NS'
    c='Tata Motors'
elif n==2:
    s='ADANIPOWER.NS'
    c='Adani Power'
choose(s)

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
# -----------------------
df=df.dropna()
#print(df.tail())

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

model.fit(X_train,y_train)


# Instead of just a 0 or 1, get the probability (confidence) of it being a 1
# .predict_proba gives back two numbers: [Probability of 0, Probability of 1]
# We only want the second number (Probability of 1), so we use [:, 1]
probs = model.predict_proba(X_test)[:, 1]

# Now, let's create a new custom prediction based on a higher threshold (e.g., 60%)
# This says: "If probability is greater than 0.6, predict 1, otherwise 0"
custom_preds = (probs > 0.6).astype(int)

new_score = precision_score(y_test, custom_preds)
print(f"New Precision Score: {new_score}")

# 1. Grab the very last row of data (most recent trading day)
last_day = X_test.iloc[-1:]

# 2. Ask the model for its confidence score that the next day is "UP"
next_day_prob = model.predict_proba(last_day)[:, 1][0]

print(f"Probability of {c} going UP next trading day: {next_day_prob:.2f}")

# Get the importance scores from the trained model
'''importances = model.feature_importances_

# Create a nice looking table to view them
feature_importance_df = pd.DataFrame({
    'Feature': predictors,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)'''

########BACKTESTING#######


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
plt.plot(test_with_preds['Buy_and_Hold'], label=f'Buy and Hold ({c})', color='grey', alpha=0.5)
plt.plot(test_with_preds['Strategy_Equity'], label='Your Strategy', color='green', linewidth=2)
plt.title('Strategy Performance vs Buy & Hold (2024-2025)')
plt.legend()
plt.show()