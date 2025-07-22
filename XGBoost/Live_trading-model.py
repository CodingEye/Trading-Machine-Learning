import MetaTrader5 as mt
import pandas as pd
import joblib
import time
from datetime import datetime, timedelta
from ta.momentum import StochasticOscillator
from ta.trend import WMAIndicator

# --- Load trained model and feature list ---
model = joblib.load('XGBoost/market_data/ustec_rf_model.joblib')
features = [
    'LWMA_15', 'LWMA_60', 'LWMA_200', 'STOCH_K', 'STOCH_D',
    'LWMA_15_LWMA_60', 'LWMA_15_LWMA_200', 'LWMA_60_LWMA_200',
    'LWMA_15_Slope', 'LWMA_60_Slope', 'LWMA_200_Slope',
    'LWMA_15_LWMA_60_Crossover', 'LWMA_15_LWMA_200_Crossover'
]

# --- Connect to MetaTrader 5 ---
mt.initialize()

# get account info
account_info = mt.account_info()
login_number = account_info.login
balance = account_info.balance
equity = account_info.equity

print()
print('login: ', login_number)
print('balance: ', balance)
print('equity: ', equity)

symbol = "USTEC"
timeframe = mt.TIMEFRAME_M5  # or your preferred timeframe

while True:
    print(f"\n[{datetime.now()}] Checking for new signal...")

    # --- Get latest 300 bars for indicator calculation ---
    rates = mt.copy_rates_from(symbol, timeframe, datetime.now() - timedelta(days=2), 300)
    df = pd.DataFrame(rates)

    print("Raw MT5 DataFrame shape:", df.shape)
    print("First 5 rows of raw data:\n", df.head())

    if len(df) < 200:
        print("Not enough bars to calculate all indicators. Waiting for more data...")
        time.sleep(60)
        continue

    # Calculate indicators
    df['LWMA_15'] = WMAIndicator(close=df['close'], window=15).wma()
    df['LWMA_60'] = WMAIndicator(close=df['close'], window=60).wma()
    df['LWMA_200'] = WMAIndicator(close=df['close'], window=200).wma()

    stoch = StochasticOscillator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14,
        smooth_window=3
    )
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()

    # Feature engineering
    df['LWMA_15_LWMA_60'] = df['LWMA_15'] - df['LWMA_60']
    df['LWMA_15_LWMA_200'] = df['LWMA_15'] - df['LWMA_200']
    df['LWMA_60_LWMA_200'] = df['LWMA_60'] - df['LWMA_200']

    df['LWMA_15_Slope'] = df['LWMA_15'].diff()
    df['LWMA_60_Slope'] = df['LWMA_60'].diff()
    df['LWMA_200_Slope'] = df['LWMA_200'].diff()

    df['LWMA_15_LWMA_60_Crossover'] = (df['LWMA_15'] > df['LWMA_60']).astype(int)
    df['LWMA_15_LWMA_200_Crossover'] = (df['LWMA_15'] > df['LWMA_200']).astype(int)

    df = df.dropna()

    if df.empty:
        print("No data left after indicator calculation. Waiting for more data...")
        time.sleep(60)
        continue

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Missing features in live data: {missing}")
        time.sleep(60)
        continue

    X_live = df[features].iloc[[-1]]
    signal = model.predict(X_live)[0]

    if signal == 1:
        print("Buy signal")
        # Place buy order here if desired
    elif signal == 0:
        print("Sell signal")
        # Place sell order here if desired
    else:
        print("No action")

    # Wait 1 minute before next check
    time.sleep(60)

# mt.shutdown()  # Not called in loop; will be called when script is stopped