# 📚 Algorithm Overview for Intraday Financial Trading

This document outlines key machine learning and statistical algorithms for modeling high-frequency intraday trading data (e.g., M5, M15, H1 candles). Each algorithm includes rationale, suitability, pros/cons, and implementation notes.

---

## 1. 🔁 LSTM (Long Short-Term Memory Networks)
**Why**: LSTMs are ideal for time-series forecasting, capturing both short- and long-term dependencies across candles and timeframes.

**Suitability**: Best suited for noisy, sequential financial data like M5 candles.

**Pros**:
- Captures temporal structure (memory of past candles)
- Accepts multi-timeframe inputs
- Flexible for classification or regression

**Cons**:
- Computationally heavy
- Overfitting risk without proper tuning

**Implementation**:
- Input features: OHLC, TICKVOL, RSI, MACD, returns
- Normalize input (Min-Max, Z-score)
- Sequence length: 50–100 (i.e., 250–500 minutes)
- Combine with M15 and H1 indicators

---

## 2. 🌲 XGBoost / LightGBM (Gradient Boosted Trees)
**Why**: Robust for tabular data and technical indicator-based features.

**Suitability**: Ideal for fast experimentation and clear feature importance.

**Pros**:
- Fast to train and interpret
- Handles feature interactions well
- Works with limited data

**Cons**:
- Needs careful feature engineering
- Not suitable for raw time-series without transformation

**Implementation**:
- Create lag-based and indicator features
- Multi-timeframe features (e.g., H1 SMA, M15 RSI)
- Label target: `CLOSE(t+1) > CLOSE(t)` (binary)

---

## 3. 📈 Temporal Convolutional Networks (TCNs)
**Why**: Use causal convolutions to learn sequence patterns more efficiently than RNNs.

**Suitability**: Good middle-ground between LSTMs and simple MLPs in terms of performance and speed.

**Pros**:
- Faster than LSTM/RNN
- Handles long-term dependencies via dilation

**Cons**:
- Slightly less intuitive for financial patterns
- Similar preprocessing demands as LSTM

**Implementation**:
- Input: Normalized OHLC, indicators
- Sequence windows (same as LSTM)
- Use dilated convolution layers

---

## 4. ✨ Transformer-Based Models (Time Series Transformers)
**Why**: Attention mechanism allows long-range dependency modeling and multi-feature relationships.

**Suitability**: Best for large-scale time-series and multi-timeframe fusion.

**Pros**:
- Captures rich relationships across time and features
- Flexible architecture

**Cons**:
- High computational resource usage
- Overfitting risk in small/noisy datasets

**Implementation**:
- Frameworks: PyTorch, HuggingFace, Darts
- Preprocess to fixed-length sequences (OHLC + indicators)
- Add time encoding / position embedding

---

## 5. 📉 ARIMA / GARCH (Statistical Models)
**Why**: Traditional baselines; useful for volatility and return modeling

**Suitability**: Best as benchmark models or in hybrid setups

**Pros**:
- Lightweight and interpretable
- GARCH is good for volatility forecasts

**Cons**:
- Not good for directional price prediction
- Requires stationarity assumptions

**Implementation**:
- ARIMA on returns or log-returns
- GARCH for volatility (e.g., HIGH-LOW range)

---

## 6. 🔄 Random Forests / Decision Trees
**Why**: Easy-to-use tree-based models for classification tasks with engineered features.

**Suitability**: Works for small/medium datasets with engineered signals

**Pros**:
- Low complexity, high interpretability
- No need for data normalization

**Cons**:
- Prone to overfitting without tuning
- Cannot model temporal dependencies directly

**Implementation**:
- Feature set: RSI, MACD, moving averages, returns
- Use majority vote for classification or ensemble

---

## 7. 📉 Logistic Regression / SVM
**Why**: Lightweight baselines for directional price movement classification

**Suitability**: Useful for benchmarking or when feature set is small and linear

**Pros**:
- Simple and fast
- Works with few features

**Cons**:
- Assumes linear boundaries
- Poor with complex non-linear market patterns

**Implementation**:
- Use with normalized or PCA-reduced features
- Target: price up/down next candle

---

## Summary Table
| Model Type | Complexity | Sequential? | Feature Engineering | Suitable For |
|------------|------------|-------------|----------------------|----------------|
| LSTM | High | ✅ Yes | Medium | Sequential, noisy data |
| XGBoost | Medium | ❌ No | High | Tabular, fast prediction |
| TCN | Medium | ✅ Yes | Medium | Pattern recognition |
| Transformer | Very High | ✅ Yes | High | Long-range patterns |
| ARIMA/GARCH | Low | ✅ (AR) | Low | Volatility, baselines |
| Random Forest | Low | ❌ No | High | Tabular classification |
| Logistic/SVM | Low | ❌ No | Low | Simple trends |

---

👉 **Next Steps**:
- [Data Processing Guide](./2-Data-Processing.md)
- [Backtesting Guide](./3-Backtesting.md)

