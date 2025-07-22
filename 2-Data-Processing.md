
Here's the content for `Data_Processing.txt`, already added earlier as a structured document. Below is a clean version you can copy directly to your file or include in your repo.

---

### 📄 `Data_Processing.txt`

````markdown
# 🧼 Data Processing for Intraday Trading ML

This file documents the end-to-end data processing pipeline to prepare intraday financial time-series (M5, M15, H1) data for use in machine learning models.

---

## 📥 1. Data Ingestion

**Sources**:
- MetaTrader 5 (via Python API)
- CSV/Parquet files of historical OHLC data

**Timeframes**:
- Primary: M5
- Secondary: M15, H1 (used for multi-timeframe features)

---

## 🔄 2. Resampling & Time Alignment

**Goal**: Sync M15/H1 indicators with M5 candles for modeling.

```python
df_15 = df.resample('15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'tickvol': 'sum'
})

df_h1 = df.resample('1H').agg({...})

df_merged = m5_df.join(df_15, on='timestamp', rsuffix='_15').join(df_h1, on='timestamp', rsuffix='_h1')
````

---

## 🧪 3. Feature Engineering

### 🔢 Raw Features:

* OHLCV: open, high, low, close, tick volume
* Returns: `log(CLOSE_t / CLOSE_{t-1})`

### 📈 Technical Indicators:

* RSI (14)
* MACD (12,26,9)
* Bollinger Bands (20)
* EMA/SMA (e.g., EMA\_10, SMA\_50)
* Stochastic Oscillator

> Tools: `ta`, `btalib`, `pandas_ta`

### 🧩 Multi-Timeframe Features:

* M15 indicators: `RSI_15`, `SMA_15`, `MACD_15`
* H1 indicators: `SMA_H1`, `VOL_H1`, etc.

### ⏱ Lag Features:

* `close_lag1`, `close_lag5`, etc.
* Lagged RSI, MACD, returns, volume

### 🔀 Price Action Features:

* Candle body: `abs(close - open)`
* Wick size: `high - max(open, close)`
* Doji indicator: `abs(open - close) / (high - low) < 0.1`
* Volatility windows: `rolling_std(close, window=10)`

---

## ⚙️ 4. Preprocessing Steps

### 🧹 Cleaning:

* Forward fill missing values
* Drop non-trading hours (optional)

### 📊 Normalization (for models like LSTM/TCN):

* **Min-Max Scaling**:

```python
(df - df.min()) / (df.max() - df.min())
```

* **Z-score Normalization**:

```python
(df - df.mean()) / df.std()
```

### 🧱 Windowing (for sequence models):

```python
def create_sequences(df, window):
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df.iloc[i:i+window])
        y.append(target[i+window])
    return np.array(X), np.array(y)
```

---

## 🎯 5. Target Construction

### 📍 Classification:

* Binary: `1` if `CLOSE(t+1) > CLOSE(t)` else `0`
* Optional: multi-class based on threshold (e.g., strong up, flat, strong down)

### 📈 Regression:

* Predict next return: `return_t+1`
* Predict volatility or candle range

---

## 📁 Final Output

* Format: CSV, NumPy arrays, or Tensor datasets
* Columns:

  * Features (indicators, returns, multi-timeframe features)
  * Target labels
  * Timestamp/index

---

## 🧠 Notes

* Always align multi-timeframe features properly to avoid data leakage.
* Normalize only **training data** independently from test/validation.
* Save processing code as scripts (`features_engineering.py`, `normalize.py`) for reuse.

---

🔗 **Next**: [Backtesting Guide](./3-Backtesting.md)

