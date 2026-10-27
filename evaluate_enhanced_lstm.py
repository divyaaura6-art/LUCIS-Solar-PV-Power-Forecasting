
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

CAPACITY = 5.0
LOOKBACK = 168
HORIZON = 24

def calendar_features(times):
    hour = times.hour.to_numpy()
    day = times.dayofyear.to_numpy()

    return np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * (day - 1) / 365.25),
        np.cos(2 * np.pi * (day - 1) / 365.25)
    ], axis=-1).astype(np.float32)

X = np.load(DATA / "X_test.npy")
Y = np.load(DATA / "y_test.npy") * CAPACITY

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

# Evaluate only midnight-issued daily forecasts.
mask = (timestamps["forecast_start"].dt.hour == 0).to_numpy()

X = X[mask]
Y = Y[mask]
starts = timestamps.loc[
    mask, "forecast_start"
].reset_index(drop=True)

past_cal = []
future_cal = []

for start in starts:
    past_times = pd.date_range(
        start - pd.Timedelta(hours=LOOKBACK),
        periods=LOOKBACK,
        freq="h"
    )
    future_times = pd.date_range(
        start, periods=HORIZON, freq="h"
    )

    past_cal.append(calendar_features(past_times))
    future_cal.append(calendar_features(future_times))

past_cal = np.asarray(past_cal, dtype=np.float32)
future_cal = np.asarray(future_cal, dtype=np.float32)

X_enhanced = np.concatenate([X, past_cal], axis=-1)
F = future_cal.reshape(len(X), -1)

model = tf.keras.models.load_model(
    BASE / "models" / "enhanced_lstm_best.keras"
)

predicted = np.clip(
    model.predict(
        [X_enhanced, F],
        batch_size=128,
        verbose=0
    ) * CAPACITY,
    0, CAPACITY
)

def metrics(actual, pred):
    return (
        mean_absolute_error(actual.ravel(), pred.ravel()),
        np.sqrt(mean_squared_error(
            actual.ravel(), pred.ravel()
        )),
        r2_score(actual.ravel(), pred.ravel())
    )

mae, rmse, r2 = metrics(Y, predicted)

# Approximate daylight hours: 06:00–18:00 IST.
day_mask = np.zeros(24, dtype=bool)
day_mask[6:19] = True

d_mae, d_rmse, d_r2 = metrics(
    Y[:, day_mask],
    predicted[:, day_mask]
)

print("\n=== ENHANCED LSTM DAILY TEST RESULTS ===")
print("Daily forecast origins:", len(X))
print(f"Overall MAE : {mae:.4f} MW")
print(f"Overall RMSE: {rmse:.4f} MW")
print(f"Overall R2  : {r2:.4f}")

print("\n=== DAYLIGHT-ONLY RESULTS ===")
print(f"Day MAE     : {d_mae:.4f} MW")
print(f"Day RMSE    : {d_rmse:.4f} MW")
print(f"Day R2      : {d_r2:.4f}")

pd.DataFrame([{
    "model": "Enhanced LSTM",
    "mae_mw": mae,
    "rmse_mw": rmse,
    "r2": r2,
    "day_mae_mw": d_mae,
    "day_rmse_mw": d_rmse,
    "day_r2": d_r2
}]).to_csv(
    OUT / "enhanced_lstm_daily_metrics.csv",
    index=False
)

print("\nSaved enhanced LSTM test metrics.")
