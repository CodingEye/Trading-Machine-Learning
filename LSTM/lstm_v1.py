import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

# Load M5 data
df = pd.read_csv("USTEC_M5.csv")
df['TIME'] = pd.to_datetime(df['TIME'])

# Feature engineering
df['Returns'] = df['CLOSE'].pct_change()
df['RSI'] = RSIIndicator(df['CLOSE'], window=14).rsi()
df['MACD'] = MACD(df['CLOSE']).macd()
df['SMA20'] = SMAIndicator(df['CLOSE'], window=20).sma_indicator()

# Assume M15/H1 features are added (e.g., from other CSVs)
# Example: df['M15_RSI'] = ... (load and align M15 RSI)

# Define target: 1 if next close > current close, else 0
df['Target'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)

# Drop NaN and select features
df = df.dropna()
features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICKVOL', 'RSI', 'MACD', 'SMA20']
X = df[features].values
y = df['Target'].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences (50 candles)
sequence_length = 50
X_seq, y_seq = [], []
for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Split data (80% train, 20% test)
train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict
predictions = (model.predict(X_test) > 0.5).astype(int)

# Evaluate
from sklearn.metrics import accuracy_score, precision_score
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))