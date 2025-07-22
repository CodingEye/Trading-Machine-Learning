import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import json
import os

def load_credentials():
    try:
        with open('config.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("config.json not found. Please create it with your MT5 credentials")
        return None

def initialize_mt5():
    credentials = load_credentials()
    if not credentials:
        return False
        
    if not mt5.initialize():
        print("MT5 initialization failed")
        return False

    authorized = mt5.login(
        credentials["login"],
        credentials["password"], 
        credentials["server"]
    )
    
    if not authorized:
        print(f"Login failed: {mt5.last_error()}")
        return False

    print("MT5 initialized and login successful")
    return True

def download_multi_timeframe_data(symbol, timeframes, start_date, end_date, base_path):
    # MT5 timeframe mapping
    mt5_timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'Daily': mt5.TIMEFRAME_D1,
        'Weekly': mt5.TIMEFRAME_W1,
        'Monthly': mt5.TIMEFRAME_MN1
    }

    for tf in timeframes:
        try:
            # Get MT5 timeframe value
            tf_value = mt5_timeframes.get(tf)
            if tf_value is None:
                print(f"Invalid timeframe: {tf}")
                continue

            # Download data
            rates = mt5.copy_rates_range(symbol, tf_value, start_date, end_date)
            if rates is None:
                print(f"Failed to download {tf} data for {symbol}")
                continue

            # Convert to DataFrame with correct column names
            df = pd.DataFrame(rates)
            
            # Convert time to datetime and set as index
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match expected format
            column_mapping = {
                'time': 'TIME',
                'open': 'OPEN',
                'high': 'HIGH',
                'low': 'LOW',
                'close': 'CLOSE',
                'tick_volume': 'TICKVOL',
                'spread': 'SPREAD',
                'real_volume': 'VOLUME'
            }
            
            # Rename columns
            df.rename(columns=column_mapping, inplace=True)
            
            # Save to CSV directly in root folder
            filename = f"{symbol}_{tf}.csv"
            filepath = os.path.join(base_path,"market_data", filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} records of {tf} data for {symbol}")
            
        except Exception as e:
            print(f"Error downloading {tf} data for {symbol}: {e}")
            print(f"MT5 Error: {mt5.last_error()}")
            
                        
def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Define symbols and timeframes
    symbols = ["USTEC", "US30", "XAUUSD"]
    timeframes = ["M1","M5","M15","H1", "Daily"]  # Can be easily modified to add/remove timeframes
    
    start_date = datetime(2025, 6, 1)
    end_date = datetime.now()

    if not initialize_mt5():
        return

    print("\nDownloading historical data...")
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        download_multi_timeframe_data(symbol, timeframes, start_date, end_date, base_path)


    # Shutdown MT5
    mt5.shutdown()
    print("\nData collection complete")

    # Print the files that should be processed by the ML framework
    print("\nFiles to process for ML framework:")
    files_to_process = []
    for symbol in symbols:
        base_name = symbol.split('.')[0]  # Remove .cash suffix
        for tf in ['Daily', 'H1', 'M15']:
            #filename = f"{base_name}_{tf}_{start_date.strftime('%Y%m%d%H%M')}_{end_date.strftime('%Y%m%d%H%M')}.csv"
            filename = f"{base_name}_{tf}.csv"
            files_to_process.append(filename)
    
    print("\nfiles_to_process = [")
    for file in files_to_process:
        print(f'    "{file}",')
    print("]")

if __name__ == "__main__":
    main()