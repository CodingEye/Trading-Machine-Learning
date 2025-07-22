import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load and clean CSV data ---
df = pd.read_csv('USTEC_historical.csv')

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Inspect basic info and check for missing values
print(df.info())
print("Missing values per column:\n", df.isna().sum())

# For simplicity, drop any rows with missing data
df.dropna(inplace=True)

# --- Step 2: Define a function to calculate LWMA ---
def lwma(series, window):
    """
    Calculate the Linear Weighted Moving Average (LWMA) for a given series.
    Weights: oldest value=1, next=2, ..., most recent=value 'window'.
    """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

# --- Step 3: Calculate LWMAs on the 'close' price column ---
df['LWMA15'] = lwma(df['close'], 15)  # Changed 'Close' to 'close'
df['LWMA60'] = lwma(df['close'], 60)
df['LWMA200'] = lwma(df['close'], 200)

# --- Step 4: Visualize the results ---
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['close'], label='Close Price', color='black')
plt.plot(df.index, df['LWMA15'], label='LWMA15', linestyle='--')
plt.plot(df.index, df['LWMA60'], label='LWMA60', linestyle='--')
plt.plot(df.index, df['LWMA200'], label='LWMA200', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('5-Minute Candle Close Price and LWMAs')
plt.legend()
plt.grid(True)
plt.show()

# --- Step 5: Save the dataframe with calculated LWMAs ---
df.to_csv('USTEC_historical_with_LWMAs.csv')
