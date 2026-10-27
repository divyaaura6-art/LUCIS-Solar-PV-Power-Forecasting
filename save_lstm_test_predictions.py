
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"

# Load the exact test windows prepared earlier
X_test = np.load(DATA / "X_test.npy")
y_test = np.load(DATA / "y_test.npy")

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv"
)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Timestamp columns:", timestamps.columns.tolist())

# Find forecast-start timestamp column
timestamp_col = timestamps.columns[0]
forecast_starts = pd.to_datetime(timestamps[timestamp_col])

# Select midnight-issued forecasts only
midnight_idx = np.where(
    (forecast_starts.dt.hour == 0) &
    (forecast_starts.dt.minute == 0)
)[0]

print("Midnight forecast days:", len(midnight_idx))

assert len(midnight_idx) == 243, (
    "Expected 243 midnight forecasts. "
    "Check timestamp alignment before proceeding."
)

model = load_model(
    BASE / "models" / "lstm_best.keras",
    compile=False
)

X_daily = X_test[midnight_idx]
y_daily = y_test[midnight_idx]

pred_daily = model.predict(
    X_daily,
    batch_size=64,
    verbose=0
)

# Convert normalized output to MW
actual_mw = y_daily * 5.0
pred_mw = np.clip(pred_daily * 5.0, 0, 5)

records = []

for i, idx in enumerate(midnight_idx):
    start = forecast_starts.iloc[idx]

    for hour in range(24):
        records.append({
            "Timestamp": start + pd.Timedelta(hours=hour),
            "Actual_MW": float(actual_mw[i, hour]),
            "LSTM_MW": float(pred_mw[i, hour])
        })

result = pd.DataFrame(records)

result.to_csv(
    OUT / "lstm_daily_predictions.csv",
    index=False
)

print("\nSaved:", OUT / "lstm_daily_predictions.csv")

print("\nLSTM DAILY TEST METRICS")
print("MAE:", mean_absolute_error(
    result["Actual_MW"], result["LSTM_MW"]
))
print("RMSE:", np.sqrt(mean_squared_error(
    result["Actual_MW"], result["LSTM_MW"]
)))
print("R2:", r2_score(
    result["Actual_MW"], result["LSTM_MW"]
))
