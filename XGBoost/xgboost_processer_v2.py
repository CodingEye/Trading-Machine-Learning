import os
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import WMAIndicator
'''Now, all indicator-enhanced data for all symbols and timeframes will be stored in a 
single file: all_indicators.csv with a SYMBOL column) in your market_data directory. 
This makes it easy to analyze or train models across all your data in one place. 
SYMBOL,Timeframe,Time,Open,High,Low,Close,VOLUME,TICKVOL,SPREAD,LWMA_15,LWMA_60,LWMA_200,STOCH_K,STOCH_D
'''
# --- Set up data path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "market_data")

# --- Process all *_all_timeframes.csv files in market_data
data_files = [f for f in os.listdir(data_path) if f.endswith('_all_timeframes.csv')]



# Process and store one indicator-enhanced file per symbol


for file in data_files:
    symbol = file.split('_')[0]
    print(f"Processing {file} for symbol {symbol}")
    df = pd.read_csv(os.path.join(data_path, file), parse_dates=['DateTime'])
    df = df.sort_values(["Timeframe", "DateTime"]).reset_index(drop=True)



    indicator_frames = []
    for tf in df['Timeframe'].unique():
        tf_df = df[df['Timeframe'] == tf].copy()
        if len(tf_df) < 210:
            continue
        tf_df["LWMA_15"] = WMAIndicator(close=tf_df["Close"], window=15).wma()
        tf_df["LWMA_60"] = WMAIndicator(close=tf_df["Close"], window=60).wma()
        tf_df["LWMA_200"] = WMAIndicator(close=tf_df["Close"], window=200).wma()
        stoch = StochasticOscillator(
            high=tf_df["High"],
            low=tf_df["Low"],
            close=tf_df["Close"],
            window=14,
            smooth_window=3
        )
        tf_df["STOCH_K"] = stoch.stoch()
        tf_df["STOCH_D"] = stoch.stoch_signal()
        tf_df["Symbol"] = symbol

        # --- Feature engineering: MA distances, slopes, crossovers ---
        tf_df['LWMA_15_LWMA_60'] = tf_df['LWMA_15'] - tf_df['LWMA_60']
        tf_df['LWMA_15_LWMA_200'] = tf_df['LWMA_15'] - tf_df['LWMA_200']
        tf_df['LWMA_60_LWMA_200'] = tf_df['LWMA_60'] - tf_df['LWMA_200']

        tf_df['LWMA_15_Slope'] = tf_df['LWMA_15'].diff()
        tf_df['LWMA_60_Slope'] = tf_df['LWMA_60'].diff()
        tf_df['LWMA_200_Slope'] = tf_df['LWMA_200'].diff()

        tf_df['LWMA_15_LWMA_60_Crossover'] = (tf_df['LWMA_15'] > tf_df['LWMA_60']).astype(int)
        tf_df['LWMA_15_LWMA_200_Crossover'] = (tf_df['LWMA_15'] > tf_df['LWMA_200']).astype(int)

        tf_df.dropna(inplace=True)
        indicator_frames.append(tf_df)
    if indicator_frames:
        symbol_indicators = pd.concat(indicator_frames, ignore_index=True)
        # Reorder columns as requested
        ordered_cols = [
            'Symbol', 'Timeframe', 'DateTime', 'Open', 'High', 'Low', 'Close',
            'Volume', 'TickVolume', 'Spread',
            'LWMA_15', 'LWMA_60', 'LWMA_200', 'STOCH_K', 'STOCH_D',
            'LWMA_15_LWMA_60', 'LWMA_15_LWMA_200', 'LWMA_60_LWMA_200',
            'LWMA_15_Slope', 'LWMA_60_Slope', 'LWMA_200_Slope',
            'LWMA_15_LWMA_60_Crossover', 'LWMA_15_LWMA_200_Crossover'
        ]
        # Add only columns that exist (in case of missing data)
        ordered_cols = [col for col in ordered_cols if col in symbol_indicators.columns]
        symbol_indicators = symbol_indicators[ordered_cols]
        out_file = os.path.join(data_path, f"{symbol}_features.csv")
        symbol_indicators.to_csv(out_file, index=False)
        print(f"Saved {len(symbol_indicators)} rows to {out_file}")
