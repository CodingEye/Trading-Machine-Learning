
# Classification vs Regression for Financial Trading (MA-based Signals)

## Overview
This guide explains the difference between classification and regression for trading strategies using moving average (MA) indicators, and why classification is usually the better fit for directional trading signals.

---

## Classification (binary:logistic)
**Use Case:** Predicts price direction (up/down) for trading decisions.

**Why Use Classification?**
- **Directly fits trading logic:** MA crossovers, slopes, and distances are directional signals (e.g., LWMA15 > LWMA60 = bullish).
- **Simplifies decision-making:** Outputs a clear buy/sell signal (1/0).
- **Robust to noise:** Focuses on trend, not exact price, so less sensitive to volatility.
- **Feature alignment:** MA-based features (distances, slopes, crossovers) are naturally binary.

**Example Target:**
```python
df['Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()  # Drop last row with NaN target
```

---

## Regression (reg:squarederror)
**Use Case:** Predicts the next price (e.g., NextClose) for strategies needing exact price levels.

**Why Not Regression for Directional Trading?**
- **Mismatch with directional focus:** Predicts price, not direction. You must post-process to get up/down signals.
- **Sensitive to noise:** Small price errors can flip the signal.
- **Feature mismatch:** MA-based features are more about direction than price level.

**Example Target:**
```python
df['NextClose'] = df['Close'].shift(-1)
```

---

## Implementation: XGBoost Classification Example

### 1. Feature Engineering
```python
# MA Distances
df['LWMA_15_LWMA_60'] = df['LWMA_15'] - df['LWMA_60']
df['LWMA_15_LWMA_200'] = df['LWMA_15'] - df['LWMA_200']
df['LWMA_60_LWMA_200'] = df['LWMA_60'] - df['LWMA_200']

# MA Slopes
df['LWMA_15_Slope'] = df['LWMA_15'].diff()
df['LWMA_60_Slope'] = df['LWMA_60'].diff()
df['LWMA_200_Slope'] = df['LWMA_200'].diff()

# MA Crossovers
df['LWMA_15_LWMA_60_Crossover'] = (df['LWMA_15'] > df['LWMA_60']).astype(int)
df['LWMA_15_LWMA_200_Crossover'] = (df['LWMA_15'] > df['LWMA_200']).astype(int)
```

### 2. Prepare Data
Remove columns not needed for training:
```python
df = df.drop(['Volume', 'Symbol', 'Timeframe'], axis=1)
```

### 3. Split Data Chronologically
```python
df['DateTime'] = pd.to_datetime(df['DateTime'])
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
train = df[:train_size]
val = df[train_size:train_size + val_size]
test = df[train_size + val_size:]

X_train = train.drop(['DateTime', 'Direction'], axis=1)
y_train = train['Direction']
X_val = val.drop(['DateTime', 'Direction'], axis=1)
y_val = val['Direction']
X_test = test.drop(['DateTime', 'Direction'], axis=1)
y_test = test['Direction']
```

### 4. Train and Evaluate XGBoost Classifier
```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
xgb.plot_importance(model)
```

---

## Why Classification is Better for MA-Based Trading
- **MA Distances:** Positive values (e.g., LWMA15 > LWMA60) = bullish, negative = bearish.
- **MA Slopes:** Positive slope = upward momentum.
- **MA Crossovers:** Classic buy/sell signals.
- **Stochastic Indicators:** Complement direction (overbought/oversold).

---

## When to Use Regression
Use regression only if you need to predict exact price levels (e.g., for take-profit/stop-loss). For most MA-based directional strategies, classification is simpler and more robust.

---

## Final Notes
- Tune XGBoost hyperparameters for best results.
- For multiple timeframes, consider separate models or feature engineering.
- For visualizations (e.g., crossovers), use matplotlib or Chart.js.

---