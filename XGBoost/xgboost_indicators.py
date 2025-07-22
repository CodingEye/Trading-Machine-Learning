import os
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import WMAIndicator

# --- Set up data path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "market_data")
csv_file = os.path.join(data_path, "USTEC_M15.csv")

# --- Load your data
df = pd.read_csv(csv_file, parse_dates=['TIME'])
df = df.sort_values("TIME")

# --- Calculate LWMA (WMA in `ta` library is LWMA)
df["LWMA_15"] = WMAIndicator(close=df["CLOSE"], window=15).wma()
df["LWMA_60"] = WMAIndicator(close=df["CLOSE"], window=60).wma()
df["LWMA_200"] = WMAIndicator(close=df["CLOSE"], window=200).wma()

# --- Calculate Stochastic Oscillator (14,3,3)
stoch = StochasticOscillator(
    high=df["HIGH"],
    low=df["LOW"],
    close=df["CLOSE"],
    window=14,          # %K length
    smooth_window=3     # %D smoothing
)
df["STOCH_K"] = stoch.stoch()
df["STOCH_D"] = stoch.stoch_signal()

# --- Drop rows with NaN values (indicator warm-up)
df.dropna(inplace=True)

# --- Preview result
print(df[["TIME", "CLOSE", "LWMA_15", "LWMA_60", "LWMA_200", "STOCH_K", "STOCH_D"]].tail())

# --- Save enhanced file (optional)
df.to_csv(os.path.join(data_path, "USTEC_M15_indicators.csv"), index=False)
