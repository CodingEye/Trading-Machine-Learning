"""
vps2
v5_scaler10 - latest - TO TEST
v5_scaler - multi scaler, added has_open_position(from v6), added ma2, expanded trade report, updated settings.
scaler - add scaler to non-binary/larger numeric
alt - alternate buy/sell/buy/sell during warmup to get even distribution
v5_random_s3alt    - simpler, 3 features for testing/understanding & extra explain/debugs
change to random warmup trades
changed ema to an lwma (but literal/var still says ema)

add claude buy/sell prob%
added claude explainers

set env vars first, before running, so that your un/pw are not in script
windoze pc;
set MT5pyLOGIN=333333
set MT5pyPASSWORD=pass

linux pc (even if python in wine etc);
export MT5pyLOGIN="333333"
export MT5pyPASSWORD="pass"
"""
###############################################################################
# Block 1 : Imports
###############################################################################
import sys, time, os, json, joblib, random
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import traceback

if sys.version_info >= (3, 12):
    print(f"{datetime.now()}: WARNING – MetaTrader5 wheels exist only for "
          f"Python ≤ 3.11; you are on {sys.version.split()[0]}", flush=True)

###############################################################################
# Block 1.5 : check commandline params... expect MAGIC as 1
###############################################################################
# expect a MAGIC# passed as a param to this python script
if len(sys.argv) > 1:
    try:
        MAGIC = int(sys.argv[1])
        print(f"Using MAGIC from command line: {MAGIC}", flush=True)
    except ValueError:
        print(f"Warning: Invalid MAGIC number '{sys.argv[1]}' provided. Using default.", flush=True)
        MAGIC = 709 # Fallback to your default if provided argument is not an int
else:
    print("No MAGIC number provided via command line. Using default.", flush=True)
    MAGIC = 709 # Your default MAGIC number if no argument is given

###############################################################################
# Block 2 : User settings
###############################################################################
#LOGIN, PASSWORD, SERVER = 333333, "passs", "server-Demo"
LOGIN          = int(os.getenv("MT5pyLOGIN"))   # grab login mt5 un from env var
PASSWORD       = os.getenv("MT5pyPASSWORD")     # get mt5 pwd from env var
SERVER         = "GlobalPrime-Demo"             # mt5 server - change!

SYMBOL         = "GER30"                 # Trading instrument
UNIQUE         = "xgb"              # unique. if you want multiple runnig on same symbol
#MAGIC          = 709                      # magic# passed as a commandline python param#1 now

NEWBARCHECKSECONDS = 1                   # number of seconds to wait before checking new bar
TF_ENTRY       = mt5.TIMEFRAME_M1         # Timeframe for entry signals
TF_TREND       = mt5.TIMEFRAME_M5         # Timeframe for trend confirmation
EMA_PERIOD     = 17                       # Period for the Exponential Moving Average
MA2_PERIOD     = 50                       # Period for the 2nd MA on TF_ENTRY. see ma2 examples
RSI_PERIOD     = 14                       # Period for the Relative Strength Index
LOT_SIZE       = 0.1                      # Trade volume in lots
DEVIATION      = 20                       # trade price price deviation
RR_RATIO       = 2.0                      # was 2, then 1, now 0.5 for quick testing. Risk/Reward ratio for TP
SL_ATR_MULTI   = 1.0                      # was 2. now 1 for quicker results/testing. Multiplier for ATR-based Stop Loss
MINIMUM_SL_POINTS = 10                    # was 10, now 5 for quick testing. min SL
THRESHOLD       = 0.40    # 0.6 Trade only if P(win) >= 60 %
WARMUP_TRADES   = 4      # 50 First 30 trades are always executed to gather data
#BASE_DIR   = r"C:\Users\"
BASE_DIR   = "C:\\pythonmt5\\"
SYMBOL_DIR = os.path.join(BASE_DIR, SYMBOL, f"{UNIQUE}_{MAGIC}")
os.makedirs(SYMBOL_DIR, exist_ok=True)
FEATURE_FILE = os.path.join(SYMBOL_DIR, "features.csv") # Switched to CSV for easier handling
# Changed to two model files
BUY_MODEL_FILE   = os.path.join(SYMBOL_DIR, "xgb_buy_model.joblib")
SELL_MODEL_FILE  = os.path.join(SYMBOL_DIR, "xgb_sell_model.joblib")
RETRAIN_EVERY = 2      # 10 Retrain after this many closed trades


# ========================
# Betting hours (local time)
# ========================
BETTING_START = 7    # Start trading at 08:00 local time (inclusive)
BETTING_END = 21     # Stop trading at 19:00 local time (exclusive)


comment=""

###############################################################################
# Block 2.5 : Setting change tracking for audit trail
###############################################################################
SETTING_FILE = os.path.join(SYMBOL_DIR, "setting.csv")

def save_settings(old_settings: dict, new_settings: dict, trade_count: int, buy_model: Optional[xgb.XGBClassifier], sell_model: Optional[xgb.XGBClassifier]):
    """Appends old/new settings and model params to setting.csv for audit trail."""
    import csv

    buy_feature_importances = buy_model.feature_importances_.tolist() if buy_model and hasattr(buy_model, "feature_importances_") else []
    sell_feature_importances = sell_model.feature_importances_.tolist() if sell_model and hasattr(sell_model, "feature_importances_") else []
    
    row = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "trade_count": trade_count,
        "old_settings": json.dumps(old_settings),
        "new_settings": json.dumps(new_settings),
        "buy_model_feature_importances": json.dumps(buy_feature_importances),
        "sell_model_feature_importances": json.dumps(sell_feature_importances),
    }
    write_header = not os.path.isfile(SETTING_FILE)
    with open(SETTING_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def current_settings_dict():
    """Return a dict of all current key settings for tracking."""
    return {
        "LOGIN": LOGIN,
        "SERVER": SERVER,
        "SYMBOL": SYMBOL,
        "UNIQUE": UNIQUE,
        "MAGIC": MAGIC,
        "NEWBARCHECKSECONDS": NEWBARCHECKSECONDS,
        "TF_ENTRY": TF_ENTRY,
        "TF_TREND": TF_TREND,
        "EMA_PERIOD": EMA_PERIOD,
        "MA2_PERIOD": MA2_PERIOD,
        "RSI_PERIOD": RSI_PERIOD,
        "LOT_SIZE": LOT_SIZE,
        "DEVIATION": DEVIATION,
        "RR_RATIO": RR_RATIO,
        "SL_ATR_MULTI": SL_ATR_MULTI,
        "MINIMUM_SL_POINTS": MINIMUM_SL_POINTS,
        "THRESHOLD": THRESHOLD,
        "WARMUP_TRADES": WARMUP_TRADES,
        "RETRAIN_EVERY": RETRAIN_EVERY,
        "BETTING_START": BETTING_START,
        "BETTING_END": BETTING_END,
        "SYMBOL_DIR": SYMBOL_DIR,
    }

###############################################################################
# Block 3 : Strategy & feature engineering
###############################################################################
def fetch_rates(timeframe: int, bars: int = 500) -> pd.DataFrame:
    """Fetches historical data from MetaTrader 5."""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, bars)
    return pd.DataFrame(rates) if rates is not None else pd.DataFrame()

def calculate_lwma(series, period):
    """Calculate Linearly Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    
    def lwma_single(values):
        if len(values) < period:
            return np.nan
        return np.dot(values[-period:], weights) / weights.sum()
    
    return series.rolling(window=period).apply(lwma_single, raw=True)

def calculate_lwma_vectorized(df, column, period):
    """Vectorized LWMA calculation"""
    weights = np.arange(1, period + 1)
    weight_sum = weights.sum()
    
    # Create weighted sum using convolution
    weighted_sum = np.convolve(df[column], weights[::-1], mode='valid')
    
    # Pad with NaN for the initial period
    result = np.full(len(df), np.nan)
    result[period-1:] = weighted_sum / weight_sum
    
    return result

# Usage:


def indicator_pack(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds technical indicators to the dataframe."""
    if df.empty:
        return df
    
    # Exponential Moving Average (EMA)
    #df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean() # ema
    #df["ema"] = calculate_lwma(df["close"], EMA_PERIOD)    # LWMA
    df["ema"] = calculate_lwma_vectorized(df, "close", EMA_PERIOD) # LWMA more efficient
    df["ma2"] = calculate_lwma_vectorized(df, "close", MA2_PERIOD) # LWMA more efficient
    # Average True Range (ATR)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    # Average Directional Index (ADX)
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.)
    tr14     = tr.rolling(14).sum()
    plus_di  = 100 * pd.Series(plus_dm).rolling(14).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm).rolling(14).sum() / tr14
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    df["adx"] = dx.rolling(14).mean()
    # Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df

def get_trade_signal(df) -> Tuple[Optional[str], Optional[pd.Series], Optional[pd.Series]]:#, Optional[float]]:
    global comment
    """
    now random buy/sell
    Checks trading conditions and returns a signal along with filter progress.
    Now includes RSI as a filter!
    """
    # iloc -1 = last bar(=currently forming), -2=prev one (=last complete bar)
    #
    entry_df = indicator_pack(fetch_rates(TF_ENTRY, 300))
    trend_df = indicator_pack(fetch_rates(TF_TREND, 300))
    if entry_df.empty or trend_df.empty or len(entry_df) < 2:
        return None, None, None  # Return if data feed has issues
    last, prev = entry_df.iloc[-2], entry_df.iloc[-3]
    trend_last = trend_df.iloc[-2]
    #atr_median = entry_df["atr"].median()
    #rsi_min, rsi_max = 30, 70
    comment+=f"Entry(tf:{TF_ENTRY}): prevClose={prev.close:.2f}, Close={last.close:.2f}, EMA={last.ema:.2f}\n"
    comment+=f"Trend(tf:{TF_TREND}): Close={trend_last.close:.2f}, EMA={trend_last.ema:.2f}\n"
    #print(f"ATR: {last.atr:.2f} (>atrMedian check: {last.atr > atr_median})")
    #print(f"ADX: {last.adx:.2f} (>20 check: {last.adx >20})")
    #print(f"RSI: {last.rsi:.2f} (30-70 range check: {30 < last.rsi < 70})")
    # Count completed trades
    completed_trades = (df["outcome"] != -1).sum()
    if completed_trades < WARMUP_TRADES:
        # Alternating during warmup
        randomSig = "BUY" if completed_trades % 2 == 0 else "SELL"
        comment+=f"WARMUP trade #{completed_trades + 1}/{WARMUP_TRADES}: Using {randomSig} signal\n"
    else:
        # Random after warmup  
        randomSig = random.choice(["BUY", "SELL"])  # Different each time you run # This is automatically seeded with current time
        #comment+=f"POST-WARMUP: Using random {randomSig} signal\n"
        comment+=f"POST-WARMUP: \n" # calcs both buy&sell and chooses best
    #print(f"random signal: {randomSig}")
    return randomSig, last, trend_last

# define the features of the model
# Removed is_buy_signal and is_sell_signal from feature_cols for individual models
# Each model will implicitly assume its own signal type
feature_cols = ["candle_size", "candle_body","top_tail","bottom_tail",
                "is_bullish", "is_bearish", "is_doji",
                "trend_is_bullish", "trend_is_bearish", "trend_is_doji", 
                "ema_distance", "candle_above_ema",
                "ma1_above_ma2", "ma1_ma2_distance"]

def build_features(candle: pd.Series, trend_candle: pd.Series, signal_type: str = None) -> Dict[str, Any]:
    """
    Constructs a feature dictionary for a given candle.
    signal_type is now used to filter data for training, not as a feature itself.
    """
    
    return {
        "timestamp": int(candle.time),
        "candle_size": candle.high - candle.low,
        "candle_body": abs(candle.close - candle.open),
        "top_tail": candle.high - max(candle.close,candle.open),
        "bottom_tail": min(candle.close, candle.open) - candle.low,
        "is_bullish": int(candle.close > candle.open),
        "is_bearish": int(candle.close < candle.open),
        "is_doji": int(candle.close == candle.open),
        "trend_is_bullish": int(trend_candle.close > trend_candle.open),
        "trend_is_bearish": int(trend_candle.close < trend_candle.open),
        "trend_is_doji": int(trend_candle.close == trend_candle.open),
        "ema_distance": abs(candle.close - candle.ema),
        "candle_above_ema": int(candle.close > candle.ema),
        "ma1_above_ma2": int(candle.ema > candle.ma2),
        "ma1_ma2_distance": abs(candle.ma2 - candle.ema),
        # signal_type is NOT a feature for the model anymore.
        # It's used to filter data for the specific BUY/SELL model.
        "is_buy_signal": int(signal_type == "BUY") if signal_type else 0, # Keep for logging/data storage
        "is_sell_signal": int(signal_type == "SELL") if signal_type else 0, # Keep for logging/data storage
        "outcome": -1,
        "entered": 0,
        "had_signal": 0,
    }
###############################################################################
# Block 4 : Learning engine
###############################################################################
def load_dataset() -> pd.DataFrame:
    """Loads the feature dataset from a CSV file and ensures required columns exist."""
    # Add 'is_buy_signal' and 'is_sell_signal' to columns for data storage, not for features
    columns = feature_cols + ["timestamp", "outcome", "entered", "had_signal", "is_buy_signal", "is_sell_signal"]
    if not os.path.isfile(FEATURE_FILE):
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(FEATURE_FILE)
    for col in columns:
        if col not in df.columns:
            df[col] = -1 if col == "outcome" else 0 # Default for outcome
    return df
   
def save_dataset(df: pd.DataFrame):
    """Saves the entire dataset back to the CSV file."""
    df.to_csv(FEATURE_FILE, index=False)

def train_model(df: pd.DataFrame, trade_type: str) -> Optional[xgb.XGBClassifier]:
    """Trains and saves an XGBoost model for a specific trade type (BUY/SELL)."""
    
    # Filter data for the specific trade type
    if trade_type == "BUY":
        trades = df[(df.outcome.isin([0, 1])) & (df['is_buy_signal'] == 1)].copy()
        model_file = BUY_MODEL_FILE
    elif trade_type == "SELL":
        trades = df[(df.outcome.isin([0, 1])) & (df['is_sell_signal'] == 1)].copy()
        model_file = SELL_MODEL_FILE
    else:
        print(f"{datetime.now()}: Invalid trade_type '{trade_type}' for training.", flush=True)
        return None

    if len(trades) < WARMUP_TRADES:
        print(f"{datetime.now()}: Not enough completed '{trade_type}' trades ({len(trades)}) to train. Need {WARMUP_TRADES}.", flush=True)
        return None
    
    X = trades[feature_cols].copy()
    y = trades["outcome"]

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X, y)
    joblib.dump(model, model_file)
    print(f"{datetime.now()}: '{trade_type}' Model saved to {model_file}", flush=True)
    return model

def get_model(trade_type: str) -> Optional[xgb.XGBClassifier]:
    """Loads a pre-trained model for a specific trade type from disk."""
    if trade_type == "BUY":
        model_file = BUY_MODEL_FILE
    elif trade_type == "SELL":
        model_file = SELL_MODEL_FILE
    else:
        return None # Should not happen

    if os.path.isfile(model_file):
        print(f"{datetime.now()}: Loading existing '{trade_type}' model from {model_file}", flush=True)
        return joblib.load(model_file)
    print(f"{datetime.now()}: No existing '{trade_type}' model found.", flush=True)
    return None

def get_buy_sell_probabilities(buy_model: Optional[xgb.XGBClassifier], sell_model: Optional[xgb.XGBClassifier], candle, trend_candle, feature_cols):
    """Get probabilities for both BUY and SELL scenarios using respective models."""
    buy_prob = 0.5
    sell_prob = 0.5
    buy_raw_pred = None
    sell_raw_pred = None

    # Features for prediction (these are now generic, no is_buy/is_sell flags)
    current_features = build_features(candle, trend_candle) # Call without signal_type for core features
    
    # Remove is_buy_signal and is_sell_signal from features for prediction, as they are not in feature_cols
    # The models are already specialized by being trained on buy/sell data
    features_for_pred_dict = {col: current_features.get(col, 0) for col in feature_cols}
    df_for_pred = pd.DataFrame([features_for_pred_dict])

    if buy_model is not None:
        buy_prob = buy_model.predict_proba(df_for_pred)[0, 1]
        # To get the raw prediction (log-odds), we can use the inverse of the sigmoid function
        buy_raw_pred = np.log(buy_prob / (1 - buy_prob)) if buy_prob > 0 and buy_prob < 1 else None
    
    if sell_model is not None:
        sell_prob = sell_model.predict_proba(df_for_pred)[0, 1]
        # To get the raw prediction (log-odds), we can use the inverse of the sigmoid function
        sell_raw_pred = np.log(sell_prob / (1 - sell_prob)) if sell_prob > 0 and sell_prob < 1 else None
    
    return buy_prob, sell_prob, buy_raw_pred, sell_raw_pred

###############################################################################
# Block 4.5 : Simplified Reporting Functions (NEW/MODIFIED)
###############################################################################

def analyze_model_performance_report(model: xgb.XGBClassifier, feature_names: list, model_type: str) -> str:
    """Generates a brief report on model's feature importances for a specific model type."""
    report_part = "\n" + "=" * 50 + "\n"
    report_part += f"XGBOOST {model_type.upper()} MODEL FEATURE IMPORTANCE\n"
    if model is None:
        report_part += f"No {model_type} model available for feature importance analysis.\n"
        return report_part

    importances = model.feature_importances_
    feature_importance_data = []
    for i, feature_name in enumerate(feature_names):
        feature_importance_data.append({'feature': feature_name, 'importance': importances[i]})
    
    feature_importance_data.sort(key=lambda x: x['importance'], reverse=True)

    report_part += "Feature Importance Ranking (F-score):\n"
    # MODIFIED: Iterate through all features, not just top 5
    for i, item in enumerate(feature_importance_data, 1):
        report_part += f"{i:2d}. {item['feature']:20s} | {item['importance']:.4f}\n"
    report_part += "=" * 50 + "\n"
    return report_part

def explain_current_prediction_report(
    buy_model: Optional[xgb.XGBClassifier], 
    sell_model: Optional[xgb.XGBClassifier],
    features_dict_raw: Dict[str, Any], 
    feature_names: list, 
    buy_prob: float, 
    sell_prob: float,
    chosen_signal: str,
    current_price: float,
    sl_level: float,
    tp_level: float,
    threshold: float,
    chosen_model_feature_importances: Optional[Dict[str, float]] = None,
    chosen_signal_raw_prediction: Optional[float] = None
) -> str:
    """Generates a brief explanation of the current prediction."""
    report_part = "\n" + "=" * 50 + "\n"
    report_part += "CURRENT PREDICTION EXPLANATION\n"
    
    if buy_model is None and sell_model is None:
        report_part += "No models available for prediction explanation.\n"
        return report_part

    report_part += f"Current Market Price: {current_price:.2f}\n"
    report_part += f"Calculated Stop Loss: {sl_level:.2f}\n"
    report_part += f"Calculated Take Profit: {tp_level:.2f}\n"
    report_part += f"Model Threshold: {threshold:.2%}\n"
    report_part += f"Predicted BUY probability: {buy_prob:.2%}\n"
    report_part += f"Predicted SELL probability: {sell_prob:.2%}\n"
    report_part += f"Chosen Signal (based on higher prob): {chosen_signal}\n"
    
    # Note about probabilities only if models are trained and they are still identical
    if buy_model and sell_model and buy_prob == sell_prob:
        report_part += "\nNOTE: BUY and SELL probabilities are identical. This suggests the models are not yet\n"
        report_part += "learning distinct direction-specific patterns, or need more training data.\n"
        report_part += "Ensure WARMUP_TRADES/RETRAIN_EVERY are sufficiently high.\n"

    report_part += "\nCurrent Feature Values (raw):\n"
    report_part += f"  {'Feature':20s}: {'Value':15s} | {'F-score (Chosen Model)':25s}\n"
    for feature_name in feature_names:
        value = features_dict_raw.get(feature_name, 'N/A')
        f_score = chosen_model_feature_importances.get(feature_name, 'N/A') if chosen_model_feature_importances else 'N/A'
        #report_part += f"  {feature_name:20s}: {str(value):15s} | {f_score:.4f}\n"
        #report_part += f"  {feature_name:20s}: {f'{value:.2f}' if isinstance(value, (float, int)) else str(value):15s} | {f_score:.4f}\n"
        report_part += f"  {feature_name:20s}: {f'{value:.4f}' if isinstance(value, (float, int)) else str(value):15s} | {f'{f_score:.4f}' if isinstance(f_score, (float, int)) else str(f_score):25s}\n"    
    # NEW: Display raw prediction (log-odds)
    if chosen_signal_raw_prediction is not None:
        report_part += f"\nRaw Prediction (Log-Odds) for Chosen Signal: {chosen_signal_raw_prediction:.4f}\n"
        if chosen_signal == "BUY":
            prob_display = buy_prob
        else:
            prob_display = sell_prob
        report_part += f"Probability (Sigmoid of Raw Prediction): {prob_display:.2%}\n"
    
    report_part += "=" * 50 + "\n"
    return report_part


###############################################################################
# Block 5.8 : Robust Open Position Check
###############################################################################
def has_open_position(symbol, magic):
    """Check with MT5 if there's any open position for this symbol and magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False  # Treat as no position in case of error
    for pos in positions:
        if pos.magic == magic:
            return True
    return False


###############################################################################
# Block 6 : Main loop (CLOSED CANDLE ENTRY ONLY)
###############################################################################
if not mt5.initialize(server=SERVER, login=LOGIN, password=PASSWORD):
    print("s/l/p=",SERVER,LOGIN,PASSWORD)
    print(f"{datetime.now()}: MT5 initialize failed – {mt5.last_error()}", flush=True)
    sys.exit()
print(f"{datetime.now()}: Connected – {SYMBOL}", flush=True)

# NEW: Load BUY and SELL models at startup
buy_model = get_model("BUY")
sell_model = get_model("SELL")

df = load_dataset()

# Generate initial model performance reports if models exist
if buy_model is not None:
    print(f"{datetime.now()}: Loaded existing BUY model")
    comment += analyze_model_performance_report(buy_model, feature_cols, "BUY")
else:
    print(f"{datetime.now()}: No existing BUY model found, will create after warmup period")

if sell_model is not None:
    print(f"{datetime.now()}: Loaded existing SELL model")
    comment += analyze_model_performance_report(sell_model, feature_cols, "SELL")
else:
    print(f"{datetime.now()}: No existing SELL model found, will create after warmup period")


open_trades: Dict[int, Dict[str, Any]] = {} # dictionary(can list, pop etc) = open_trades[ticket]={timestamp,signal}

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# --- Track last closed candle timestamp to avoid duplicate trades per candle
last_processed_candle_time = None

while True:
    try:
        now = datetime.now()
        current_hour = now.hour

        if not (BETTING_START <= current_hour < BETTING_END):
            print(f"{now}: Outside betting hours ({BETTING_START}-{BETTING_END}), skipping trade logic.", flush=True)
            time.sleep(60)
            continue

        # 1. Fetch candles and get timestamp of last closed candle
        entry_df = indicator_pack(fetch_rates(TF_ENTRY, 300))
        if entry_df.empty or len(entry_df) < 2:
            print(f"{now}: Not enough data, retrying...", flush=True)
            time.sleep(30)
            continue
        last = entry_df.iloc[-2] # -1 = last bar(=currently forming), -2=prev one (=last complete bar)
        last_candle_time = int(last['time'])

        # 2. Run trade logic only ONCE per new closed candle
        if last_processed_candle_time is not None and last_candle_time == last_processed_candle_time:
            # No new closed candle, wait and check again
            time.sleep(NEWBARCHECKSECONDS)
            continue
        last_processed_candle_time = last_candle_time
        comment=">>"+"-" * 90+f"<<\nLast complete candle time: {datetime.utcfromtimestamp(last_candle_time).strftime('%H:%M')}\n"
        #5 check for close
        for ticket in list(open_trades):
            ''' the ticket stored in open_trades, which is returned from order_send is actually a position_id in history
            so have to get history_deals_get and then look at deal.entry ==1(out) is the closing of a trade, ==0(in)
            the out has the profit. history_orders_get doesn't have profit.
            '''
            deals = mt5.history_deals_get(position=ticket)
            print("numOpen=",len(open_trades), open_trades," ticket=",ticket," deals=",deals)
            for deal in deals:
                if deal.entry==1:
                    profit = deal.profit
                    print(ticket," profit=",profit)
                    
                    # Get both timestamp and signal type
                    trade_info = open_trades.pop(ticket) # remove trade from open_trades... pop it off
                    signal_timestamp = trade_info['timestamp']
                    signal_type = trade_info['signal_type']
                    
                    # Update the outcome AND store signal type in the row
                    mask = df['timestamp'] == signal_timestamp
                    df.loc[mask, 'outcome'] = int(profit > 0)
                    
                    print(f"{datetime.now()}: Ticket {ticket} ({signal_type}) {datetime.fromtimestamp(signal_timestamp).strftime('%H:%M')} closed. P/L: {profit:.2f}. Updating dataset.", flush=True)
                    
                    closed_trade_count = (df['outcome'] != -1).sum() # outcome is 0(false) for loss, 1(true) for win, -1 for no signal/trade
                    
                    # Retrain both models if criteria met
                    if closed_trade_count > 0 and closed_trade_count % RETRAIN_EVERY == 0:
                        print(f"{datetime.now()}: Reached {closed_trade_count} closed trades. Retraining BUY model...")
                        old_settings = current_settings_dict()
                        buy_model = train_model(df, "BUY") 
                        
                        print(f"{datetime.now()}: Retraining SELL model...")
                        sell_model = train_model(df, "SELL") 
                        
                        if buy_model and sell_model:
                            print(f"{datetime.now()}: Models successfully retrained.", flush=True)
                            new_settings = current_settings_dict()
                            save_settings(old_settings, new_settings, closed_trade_count, buy_model, sell_model)                        
                            print(f"{datetime.now()}: Settings audit logged to {SETTING_FILE}", flush=True)   
                            
                            # Generate model performance reports after retraining
                            comment += analyze_model_performance_report(buy_model, feature_cols, "BUY")
                            comment += analyze_model_performance_report(sell_model, feature_cols, "SELL")

        # 3. Now get the signal using only closed candles
        signal, candle, trend_candle = get_trade_signal(df)

        if candle is None:
            time.sleep(30)
            continue

        # Build initial features based on the signal from get_trade_signal
        # IMPORTANT: build_features no longer takes signal_type as a feature for the model,
        # but we still need it for the 'is_buy_signal'/'is_sell_signal' columns in the DataFrame.
        feat = build_features(candle, trend_candle, signal) # Pass signal for data storage

        feat["had_signal"] = int(signal is not None)

        # Enhanced Trade Decision Logic
        prob = 0.5
        buy_prob= 0.5
        sell_prob= 0.5
        buy_raw_pred = None # Initialize raw predictions
        sell_raw_pred = None
        chosen_signal = signal # Initialize chosen_signal

        # Calculate SL/TP points for the current candle, regardless of trade acceptance
        tick = mt5.symbol_info_tick(SYMBOL)
        current_price = tick.ask if signal == "BUY" else tick.bid # Price at decision point
        sl_points = max(candle.atr * SL_ATR_MULTI, MINIMUM_SL_POINTS)
        tp_points = sl_points * RR_RATIO
        
        if buy_model and sell_model:
            # Use separate models for probabilities and raw predictions
            buy_prob, sell_prob, buy_raw_pred, sell_raw_pred = get_buy_sell_probabilities(buy_model, sell_model, candle, trend_candle, feature_cols)

            comment+=f"{now}: BUY probability: {buy_prob:.2%}, SELL probability: {sell_prob:.2%}\n"

            # choose the best signal:
            best_signal = "BUY" if buy_prob > sell_prob else "SELL"
            best_prob = max(buy_prob, sell_prob)
            comment+=f"Best opportunity: {best_signal} with {best_prob:.2%} win probability\n"
            
            # Update 'signal' and 'feat' ONLY if the best_signal differs from the initial random signal
            if best_signal != signal:
                signal = best_signal
                # Rebuild features with the *chosen* best signal for correct 'is_buy_signal'/'is_sell_signal' in DataFrame
                feat = build_features(candle, trend_candle, signal)
                feat["had_signal"] = int(signal is not None)
            prob = best_prob


        # Calculate SL/TP levels based on the chosen signal
        calculated_sl_level = current_price - sl_points if signal == "BUY" else current_price + sl_points
        calculated_tp_level = current_price + tp_points if signal == "BUY" else current_price - tp_points
        
        chosen_signal = signal # Update chosen_signal based on model's choice

        # Determine chosen model's feature importances and raw prediction for reporting
        chosen_model_importances_dict = {}
        chosen_signal_raw_prediction = None
        if chosen_signal == "BUY" and buy_model:
            chosen_model_importances_dict = dict(zip(feature_cols, buy_model.feature_importances_))
            chosen_signal_raw_prediction = buy_raw_pred
        elif chosen_signal == "SELL" and sell_model:
            chosen_model_importances_dict = dict(zip(feature_cols, sell_model.feature_importances_))
            chosen_signal_raw_prediction = sell_raw_pred


        # Generate current prediction explanation report
        features_dict_raw = {col: feat.get(col, 0) for col in feature_cols}
        comment += explain_current_prediction_report(
            buy_model,
            sell_model,
            features_dict_raw, 
            feature_cols, 
            buy_prob, 
            sell_prob,
            chosen_signal,
            current_price,
            calculated_sl_level,
            calculated_tp_level,
            THRESHOLD,
            chosen_model_importances_dict,
            chosen_signal_raw_prediction
        )

        total_trades_seen = (df["outcome"] != -1).sum()
        use_filter = (buy_model is not None and sell_model is not None) and (total_trades_seen >= WARMUP_TRADES)
        accept_trade = (prob >= THRESHOLD) if use_filter else True
        comment+=f"{now}: Magic#{MAGIC} #tradesSeen={total_trades_seen} prob={prob:.2%} signal={signal} accept_trade={accept_trade}\n"

        # 4 trade if signal
        if signal and accept_trade:
            if has_open_position(SYMBOL, MAGIC):
                comment+=f"{now}: Open trade detected by MT5, skipping new trade.\n"
            else:
                reasoning = "Initial warmup period" if total_trades_seen < WARMUP_TRADES else f"Model predicted {prob:.2%} win probability."
                
                # Use the already calculated price, sl, tp levels
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": SYMBOL,
                    "volume": LOT_SIZE,
                    "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": current_price,
                    "sl": calculated_sl_level,
                    "tp": calculated_tp_level,
                    "deviation": DEVIATION,
                    "magic": MAGIC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    feat["entered"] = 1
                    open_trades[result.order] = {
                        'timestamp': feat["timestamp"],
                        'signal_type': signal
                    }
                    comment+=f"{now}: {signal} ticket={result.order} prob={prob:.2%}\n"
                    comment+=f"Reasoning: {reasoning}\n"
                else:
                    comment+=f"{now}: Order send failed, retcode={result.retcode}\n"
                
                report_file = os.path.join(SYMBOL_DIR, f"trade_report_{datetime.now().strftime('%Y%m%d_%H%M')}_{MAGIC}.txt")
                with open(report_file, 'w') as f:
                    f.write(comment)


        print(comment, flush=True)
        # NOW add the row to DataFrame after trade execution (entered flag is set correctly)
        df_new_row = pd.DataFrame([feat])
        if not df_new_row.empty and df_new_row.notna().any().any():
            # Ensure both DataFrames have the same columns
            if df.empty:
                df = df_new_row.copy()
            else:
                df = pd.concat([df, df_new_row], ignore_index=True)

        save_dataset(df)

    except Exception as e:
        print(f"{datetime.now()}: Runtime error – {e}", flush=True)
        traceback.print_exc()
    time.sleep(10)
