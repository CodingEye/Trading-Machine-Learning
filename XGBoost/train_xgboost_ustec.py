import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read all rows
df = pd.read_csv('XGBoost/market_data/USTEC_features.csv')
#df = pd.read_csv('XGBoost/market_data/USTEC_features.csv', nrows=100)  # For testing, read first 100 rows


# Feature engineering: MA distances (header uses underscores)
df['LWMA_15_LWMA_60'] = df['LWMA_15'] - df['LWMA_60']
df['LWMA_15_LWMA_200'] = df['LWMA_15'] - df['LWMA_200']
df['LWMA_60_LWMA_200'] = df['LWMA_60'] - df['LWMA_200']

# MA slopes
df['LWMA_15_Slope'] = df['LWMA_15'].diff()
df['LWMA_60_Slope'] = df['LWMA_60'].diff()
df['LWMA_200_Slope'] = df['LWMA_200'].diff()

# MA crossovers
df['LWMA_15_LWMA_60_Crossover'] = (df['LWMA_15'] > df['LWMA_60']).astype(int)
df['LWMA_15_LWMA_200_Crossover'] = (df['LWMA_15'] > df['LWMA_200']).astype(int)

# Example target: 1 if Close > Open, else 0
df['Target'] = (df['Close'] > df['Open']).astype(int)

# Drop rows with NaN (from diff)
df = df.dropna()

# Features for model (update to match new feature names)
features = [
    'LWMA_15', 'LWMA_60', 'LWMA_200', 'STOCH_K', 'STOCH_D',
    'LWMA_15_LWMA_60', 'LWMA_15_LWMA_200', 'LWMA_60_LWMA_200',
    'LWMA_15_Slope', 'LWMA_60_Slope', 'LWMA_200_Slope',
    'LWMA_15_LWMA_60_Crossover', 'LWMA_15_LWMA_200_Crossover'
]

X = df[features]
y = df['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(clf, 'XGBoost/market_data/ustec_rf_model.joblib')
print('Model saved to XGBoost/market_data/ustec_rf_model.joblib')