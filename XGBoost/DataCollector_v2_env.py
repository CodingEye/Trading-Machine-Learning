import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_mt5():
    """Initializes MT5 connection using credentials from environment variables."""
    login = os.getenv("MT5pyLOGIN")
    password = os.getenv("MT5pyPASSWORD")
    server = os.getenv("MT5SERVER")
    if not (login and password and server):
        print("MT5 credentials not found in environment variables. Please set MT5pyLOGIN, MT5pyPASSWORD, and MT5SERVER.")
        return False

    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        return False

    authorized = mt5.login( int(login), password, server )

    if not authorized:
        print(f"Login failed for account {login}: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print("MT5 initialized and login successful")
    return True

def download_multi_timeframe_data(symbol, start_date, end_date, data_path, timeframes):
    """Downloads data for multiple timeframes and saves it to the specified data_path."""
    for tf_name, tf_value in timeframes.items():
        try:
            rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"No {tf_name} data downloaded for {symbol}. MT5 Error: {mt5.last_error()}")
                continue

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            column_mapping = {
                'time': 'TIME', 'open': 'OPEN', 'high': 'HIGH', 'low': 'LOW',
                'close': 'CLOSE', 'tick_volume': 'TICKVOL', 'spread': 'SPREAD', 'real_volume': 'VOLUME'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Ensure all expected columns exist
            for col in ['TIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'SPREAD', 'VOLUME']:
                 if col not in df.columns:
                     df[col] = 0

            filename = f"{symbol}_{tf_name}.csv"
            filepath = os.path.join(data_path, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} records to {filepath}")

        except Exception as e:
            print(f"Error downloading {tf_name} data for {symbol}: {e}")
            print(f"MT5 Error: {mt5.last_error()}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "market_data")
    os.makedirs(data_path, exist_ok=True)
    print(f"Data will be saved to: {data_path}")

    # Define symbols and date range
    symbols = ["USTEC", "US30", "DE40"] #"XAUUSD"
    # Define timeframes to use
    timeframes = {
        'D1': mt5.TIMEFRAME_D1,
        'H1': mt5.TIMEFRAME_H1,
        'M30': mt5.TIMEFRAME_M30,
        'M15': mt5.TIMEFRAME_M15,
        'M5': mt5.TIMEFRAME_M5,
        'M1': mt5.TIMEFRAME_M1
    }
    start_date = datetime(2025, 1, 1)  # Using a more reasonable start date for more data
    end_date = datetime.now()

    if not initialize_mt5():
        return

    print("\nDownloading historical data...")
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        download_multi_timeframe_data(symbol, start_date, end_date, data_path, timeframes)

    mt5.shutdown()
    print("\nData collection complete. MT5 connection shut down.")

if __name__ == "__main__":
    main()