import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import ta

# Load data
df = pd.read_csv("USTEC_historical.csv", parse_dates=["time"], index_col="time")

# Calculate additional technical indicators
df["MA_15"] = df["close"].rolling(window=15).mean()
df["MA_50"] = df["close"].rolling(window=50).mean()
df["RSI"] = ta.momentum.rsi(df["close"], window=14)
df["MACD"] = ta.trend.macd_diff(df["close"])
df["BB_upper"], df["BB_middle"], df["BB_lower"] = ta.volatility.bollinger_bands(df["close"])

# Drop NaNs after indicator calculation
df.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = ["open", "high", "low", "close", "volume", 
            "MA_15", "MA_50", "RSI", "MACD", 
            "BB_upper", "BB_middle", "BB_lower"]
df_scaled = scaler.fit_transform(df[features])

# Create sequences with look_back and forecast windows
def create_sequences(data, look_back=50, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back:i + look_back + forecast_horizon, 3])  # Predicting close price
    return np.array(X), np.array(y)

# Parameters
look_back = 50
forecast_horizon = 1
X, y = create_sequences(df_scaled, look_back, forecast_horizon)

# Train-test split with proper time ordering
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build enhanced LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(look_back, len(features))),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dense(16, activation="relu"),
    Dense(forecast_horizon)
])

# Compile with proper learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train with validation split
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Save final model and scaler
model.save("nasdaq_lstm_model.h5")
np.save("scaler.npy", scaler.scale_)