
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

X_test = np.load(DATA / "X_test.npy")
y_test = np.load(DATA / "y_test.npy")

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

model = tf.keras.models.load_model(
    BASE / "models" / "lstm_best.keras"
)

pred = model.predict(X_test, batch_size=128, verbose=1)

# Convert normalized predictions back into MW.
actual = y_test * CAPACITY
predicted = np.clip(pred * CAPACITY, 0, CAPACITY)

# Overall test metrics.
mae = mean_absolute_error(actual.ravel(), predicted.ravel())
rmse = np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel()))
r2 = r2_score(actual.ravel(), predicted.ravel())

# Daylight-only metrics, based on target hours 06:00–18:00 IST.
hour_offsets = np.arange(24)
start_hours = timestamps["forecast_start"].dt.hour.to_numpy()
target_hours = (start_hours[:, None] + hour_offsets) % 24
daylight_mask = (target_hours >= 6) & (target_hours < 19)

day_mae = mean_absolute_error(
    actual[daylight_mask],
    predicted[daylight_mask]
)
day_rmse = np.sqrt(mean_squared_error(
    actual[daylight_mask],
    predicted[daylight_mask]
))
day_r2 = r2_score(
    actual[daylight_mask],
    predicted[daylight_mask]
)

print()
print("=== LUCIS LSTM TEST RESULTS ===")
print(f"Test sequences: {len(X_test)}")
print(f"Overall MAE  : {mae:.4f} MW")
print(f"Overall RMSE : {rmse:.4f} MW")
print(f"Overall R2   : {r2:.4f}")
print()
print("=== DAYLIGHT-ONLY RESULTS ===")
print(f"Day MAE      : {day_mae:.4f} MW")
print(f"Day RMSE     : {day_rmse:.4f} MW")
print(f"Day R2       : {day_r2:.4f}")

pd.DataFrame({
    "metric": [
        "overall_mae_mw",
        "overall_rmse_mw",
        "overall_r2",
        "daylight_mae_mw",
        "daylight_rmse_mw",
        "daylight_r2"
    ],
    "value": [
        mae, rmse, r2,
        day_mae, day_rmse, day_r2
    ]
}).to_csv(OUT / "lstm_test_metrics.csv", index=False)

# Save the first held-out 24-hour forecast.
first_start = timestamps["forecast_start"].iloc[0]
first_times = pd.date_range(first_start, periods=24, freq="h")

pd.DataFrame({
    "Timestamp": first_times,
    "Actual_MW": actual[0],
    "Predicted_MW": predicted[0]
}).to_csv(OUT / "lstm_first_24h.csv", index=False)

print()
print("Saved metrics and first forecast to:", OUT)
