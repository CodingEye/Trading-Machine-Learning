# gemini created, export mt5 data to csv for use in backtrader

'''
set env vars first, before running, so that your un/pw are not in script
windoze pc;
set MT5pyLOGIN=333333
set MT5pyPASSWORD="pass"

linux pc (even if python in wine etc);
export MT5pyLOGIN="333333"
export MT5pyPASSWORD="pass"
'''
import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

def export_mt5_data_to_csv(
    symbol: str,
    timeframe: int, # e.g., mt5.TIMEFRAME_M1, mt5.TIMEFRAME_H1
    start_date: datetime,
    end_date: datetime,
    output_filename: str,
    ema_period: int = 20, # Example: Period for Exponential Moving Average
    login: int = None,
    password: str = None,
    server: str = None
):
    """
    Connects to MetaTrader5, fetches historical data, calculates EMA,
    and saves it to a CSV file.

    Args:
        symbol (str): The trading symbol (e.g., "EURUSD", "GER30").
        timeframe (int): The MT5 timeframe (e.g., mt5.TIMEFRAME_M1 for 1-minute).
        start_date (datetime): The start date for data fetching.
        end_date (datetime): The end date for data fetching.
        output_filename (str): The path and name for the output CSV file.
        ema_period (int): Period for the Exponential Moving Average calculation.
        login (int, optional): MT5 account login. Defaults to None (uses default if already logged in).
        password (str, optional): MT5 account password. Defaults to None.
        server (str, optional): MT5 server name. Defaults to None.
    """
    # 1. Initialize MetaTrader 5 connection
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return

    print(f"Successfully connected to MT5. Fetching data for {symbol} ({timeframe})...")

    # 2. Request historical rates
    # mt5.copy_rates_range(symbol, timeframe, start_date, end_date) fetches data
    # within a specific date range.
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # Shutdown connection to MetaTrader 5
    mt5.shutdown()

    if rates is None:
        print(f"No data found for {symbol} in the specified range. Error: {mt5.last_error()}")
        return

    if len(rates) == 0:
        print(f"No rates returned for {symbol} in the specified range.")
        return

    # 3. Create a Pandas DataFrame from the rates
    # The 'time' column in MT5 rates is a Unix timestamp
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s') # Convert timestamp to datetime
    
    # Rename columns to be more Backtrader-friendly (lowercase)
    df.rename(columns={
        'time': 'datetime',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'spread': 'spread',
        'real_volume': 'real_volume'
    }, inplace=True)

    # Set 'datetime' as index, which is common for time series data
    df.set_index('datetime', inplace=True)

    # 4. Calculate a simple Exponential Moving Average (EMA) as an example indicator
    # Backtrader can calculate indicators itself, but you might want to pre-calculate some
    # complex ones or verify your data before feeding it.
    if 'close' in df.columns:
        df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        print(f"Calculated {ema_period}-period EMA.")
    else:
        print("Warning: 'close' column not found, cannot calculate EMA.")

    # 5. Save the DataFrame to a CSV file
    try:
        # Drop columns not typically used by Backtrader's generic CSV feed,
        # or keep them if your custom feed expects them.
        # For a simple OHLCV feed, 'spread' and 'real_volume' might be dropped.
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'ema']
        df_to_save = df[columns_to_keep].copy()

        df_to_save.to_csv(output_filename, index=True) # index=True saves the datetime index
        print(f"Data successfully exported to {output_filename}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define your MT5 account credentials (replace with your actual details or environment variables)
    # It's safer to use environment variables for sensitive info.
    # For example:
    # import os
    # mt5_login = int(os.getenv("MT5_LOGIN", "YOUR_MT5_LOGIN"))
    # mt5_password = os.getenv("MT5_PASSWORD", "YOUR_MT5_PASSWORD")
    # mt5_server = os.getenv("MT5_SERVER", "YOUR_MT5_SERVER")
    
    # For demonstration, using placeholders:
    MT5_LOGIN          = int(os.getenv("MT5pyLOGIN"))   # grab login mt5 un from env var
    MT5_PASSWORD       = os.getenv("MT5pyPASSWORD")     # get mt5 pwd from env var
    MT5_SERVER = os.getenv("MT5SERVER") # Replace with your MT5 server name (e.g., "MetaQuotes-Demo")

    # Define data parameters
    SYMBOL = "DE40"
    TIMEFRAME = mt5.TIMEFRAME_M1 # 1-minute bars
    HOUROFFSET = 3
    
    # Fetch data for the last 30 days as an example
    END_DATE = datetime.now() + timedelta(hours=HOUROFFSET)
    START_DATE = END_DATE - timedelta(days=10) 

    OUTPUT_CSV_FILE = f"{SYMBOL}_{TIMEFRAME}_{START_DATE.strftime('%Y%m%d')}_to_{END_DATE.strftime('%Y%m%d')}.csv"

    export_mt5_data_to_csv(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        output_filename=OUTPUT_CSV_FILE,
        ema_period=20,
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER
    )

    print("\n--- Example of loading the data into Backtrader (conceptual) ---")
    print("In a separate Backtrader script:")
    print("import backtrader as bt")
    print("class MyStrategy(bt.Strategy): ...")
    print("cerebro = bt.Cerebro()")
    print(f"data = bt.feeds.GenericCSVData(dataname='{OUTPUT_CSV_FILE}',")
    print("                                datetime=0, # Column index for datetime")
    print("                                open=1, high=2, low=3, close=4, volume=5,")
    print("                                openinterest=-1, # Not in this CSV")
    print("                                dtformat='%Y-%m-%d %H:%M:%S', # Adjust format if needed")
    print("                                timeframe=bt.TimeFrame.Minutes, compression=1)")
    print("cerebro.adddata(data)")
    print("cerebro.addstrategy(MyStrategy)")
    print("cerebro.run()")

