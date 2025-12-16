# Advanced-time-series-forecasting-with-neural# ============================================================
# Advanced Time Series Forecasting with Uncertainty Quantification
# LSTM + Monte Carlo Dropout vs Baseline LSTM
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------
# 1. DATA GENERATION
# ------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

N_OBS = 1000
N_FEATURES = 5
SEQ_LEN = 30

t = np.arange(N_OBS)

# Trend and regime shift
trend = 0.005 * t
regime_shift = np.where(t > 500, 2.0, 0.0)

# Seasonality
seasonal_1 = np.sin(2 * np.pi * t / 50)
seasonal_2 = np.cos(2 * np.pi * t / 100)

base_signal = trend + seasonal_1 + seasonal_2 + regime_shift

# Correlated noise
cov = 0.7 * np.ones((N_FEATURES, N_FEATURES)) + 0.3 * np.eye(N_FEATURES)
noise = np.random.multivariate_normal(
    mean=np.zeros(N_FEATURES),
    cov=cov,
    size=N_OBS
)

data = np.array([base_signal + noise[:, i] for i in range(N_FEATURES)]).T

df = pd.DataFrame(
    data,
    columns=[f"feature_{i+1}" for i in range(N_FEATURES)]
)

# ------------------------------------------------------------
# 2. DATA PREPARATION
# ------------------------------------------------------------

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])  # Predict feature_1
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LEN)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ------------------------------------------------------------
# 3. BASELINE LSTM MODEL (POINT FORECAST)
# ------------------------------------------------------------

def build_baseline_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

baseline_model = build_baseline_lstm((SEQ_LEN, N_FEATURES))
baseline_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=0
)

baseline_preds = baseline_model.predict(X_test).flatten()

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)

# ------------------------------------------------------------
# 4. PROBABILISTIC LSTM WITH MC DROPOUT
# ------------------------------------------------------------

def build_mc_dropout_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

mc_model = build_mc_dropout_lstm((SEQ_LEN, N_FEATURES))
mc_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=0
)

# ------------------------------------------------------------
# 5. MONTE CARLO DROPOUT INFERENCE
# ------------------------------------------------------------

def mc_dropout_predictions(model, X, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        preds = model(X, training=True).numpy().flatten()
        predictions.append(preds)
    predictions = np.array(predictions)

    mean_pred = predictions.mean(axis=0)

    lower_80 = np.percentile(predictions, 10, axis=0)
    upper_80 = np.percentile(predictions, 90, axis=0)

    lower_95 = np.percentile(predictions, 2.5, axis=0)
    upper_95 = np.percentile(predictions, 97.5, axis=0)

    return mean_pred, lower_80, upper_80, lower_95, upper_95

mc_mean, l80, u80, l95, u95 = mc_dropout_predictions(mc_model, X_test)

# ------------------------------------------------------------
# 6. EVALUATION METRICS
# ------------------------------------------------------------

def coverage_probability(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

def average_interval_width(lower, upper):
    return np.mean(upper - lower)

mc_rmse = np.sqrt(mean_squared_error(y_test, mc_mean))
mc_mae = mean_absolute_error(y_test, mc_mean)

coverage_80 = coverage_probability(y_test, l80, u80)
coverage_95 = coverage_probability(y_test, l95, u95)

interval_width_80 = average_interval_width(l80, u80)
interval_width_95 = average_interval_width(l95, u95)

# ------------------------------------------------------------
# 7. RESULTS SUMMARY
# ------------------------------------------------------------

results = pd.DataFrame({
    "Model": ["Baseline LSTM", "Probabilistic LSTM (MC Dropout)"],
    "RMSE": [baseline_rmse, mc_rmse],
    "MAE": [baseline_mae, mc_mae],
    "80% Coverage": [np.nan, coverage_80],
    "95% Coverage": [np.nan, coverage_95],
    "Avg Interval Width (95%)": [np.nan, interval_width_95]
})

print("\n===== MODEL COMPARISON RESULTS =====\n")
print(results)

# ------------------------------------------------------------
# END OF PROJECT
# ------------------------------------------------------------
