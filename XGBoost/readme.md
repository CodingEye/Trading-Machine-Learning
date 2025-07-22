# 🧠 Machine Learning for Intraday Financial Trading

This repository showcases machine learning models applied to high-frequency trading data, especially focused on intraday strategies using M5 (5-minute) candles with multi-timeframe context (e.g., M15, H1). It compares traditional and deep learning algorithms, discusses implementation details, and evaluates which models work best under different assumptions.

---

## 📌 Objectives

- Apply multiple machine learning algorithms to intraday trading data
- Incorporate multi-timeframe indicators (e.g., M5, M15, H1)
- Compare deep learning and traditional approaches for directional price prediction
- Provide modular and reproducible code for backtesting and modeling

---

## ⚙️ Dataset & Features

- Source: [Your data source or simulator]
- Candlestick data (OHLC, TICKVOL) at M5 resolution
- Technical Indicators: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic
- Features engineered across M5, M15, and H1 timeframes
- Targets:
  - Binary: `CLOSE(t+1) > CLOSE(t)` (Up/Down classification)
  - Regression: Return prediction or volatility estimation

---

## 🧪 Algorithms Compared

### 1. 🔁 LSTM (Long Short-Term Memory)
**Use Case**: Ideal for learning sequential patterns in price data  
**Strengths**:
- Captures temporal dependencies
- Can fuse multiple timeframes
- Supports classification & regression  
**Challenges**:
- Requires heavy preprocessing
- Prone to overfitting on noisy M5 data

**Implementation Tips**:
- Normalize OHLC and volume
- Sequence window: 50–100 candles (250–500 minutes)
- Merge M15 and H1 indicators for trend context

<!-- Code: LSTM implementation in /LSTM/lstm_v1.py -->

---

### 2. 🌲 XGBoost / LightGBM (Gradient Boosted Trees)
**Use Case**: Tabular features with technical indicators  
**Strengths**:
- Fast and interpretable
- Excellent with engineered features
- Easier to debug than deep learning

**Challenges**:
- Requires feature engineering
- Less suitable for raw sequential modeling

**Feature Ideas**:
- Lagged returns, RSI, MACD, Bollinger Bands
- Multi-timeframe signals (e.g., M15 RSI, H1 MA)

<!-- Code: XGBoost implementation in /XGBoost/train_xgboost_ustec.py -->

---

### 3. 🧠 Temporal Convolutional Networks (TCNs)
**Use Case**: Alternative to LSTMs with faster training  
**Strengths**:
- Captures long-range dependencies
- Training speed advantage over LSTM  
**Challenges**:
- Still needs good preprocessing
- Less intuitive for noisy, irregular financial patterns

<!-- Code: TCN implementation (add path if available) -->

---

### 4. ✨ Transformers for Time Series
**Use Case**: Modeling long-term attention across timeframes  
**Strengths**:
- State-of-the-art for sequence modeling
- Handles multiple timeframes effectively  
**Challenges**:
- Very resource intensive
- Needs lots of data + tuning

<!-- Code: Transformer implementation (add path if available) -->

---

### 5. 📉 ARIMA / GARCH (Statistical Models)
**Use Case**: Baselines and volatility modeling  
**Strengths**:
- Lightweight, interpretable
- GARCH is useful for volatility estimation  
**Challenges**:
- Not suited for directional prediction in M5
- Struggles with non-linear dependencies

<!-- Code: ARIMA/GARCH implementation (add path if available) -->

---

## 📊 Evaluation Strategy

| Metric | Description |
|--------|-------------|
| Accuracy | For classification models |
| F1 Score | For imbalance-aware evaluation |
| Sharpe Ratio | For profitability of predicted signals |
| MAE / RMSE | For regression (returns/volatility) |
| Drawdown | For backtest robustness |
