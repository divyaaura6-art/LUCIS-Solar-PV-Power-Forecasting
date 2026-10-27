
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

CAPACITY = 5.0

X = np.load(DATA / "X_test.npy")
Y = np.load(DATA / "y_test.npy") * CAPACITY

times = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

# One day-ahead forecast, issued at midnight.
mask = (times["forecast_start"].dt.hour == 0).to_numpy()

X = X[mask]
Y = Y[mask]
forecast_times = times.loc[mask, "forecast_start"].reset_index(drop=True)

print("Independent daily forecast origins:", len(X))

if len(X) == 0:
    raise RuntimeError("No midnight forecast windows available")

lstm = tf.keras.models.load_model(
    BASE / "models" / "lstm_best.keras"
)
rf = joblib.load(BASE / "models" / "random_forest.joblib")

predictions = {
    "LSTM": np.clip(
        lstm.predict(X, batch_size=128, verbose=0) * CAPACITY,
        0, CAPACITY
    ),
    "Random Forest": np.clip(
        rf.predict(X.reshape(-1, 168)) * CAPACITY,
        0, CAPACITY
    ),
    "Persistence 24h": X[:, -24:, 0] * CAPACITY,
    "Persistence 7d": X[:, :24, 0] * CAPACITY
}

def calculate(actual, predicted):
    return (
        mean_absolute_error(actual.ravel(), predicted.ravel()),
        np.sqrt(mean_squared_error(
            actual.ravel(), predicted.ravel()
        )),
        r2_score(actual.ravel(), predicted.ravel())
    )

# Every daily forecast starts at 00:00 IST.
day_mask = np.zeros(24, dtype=bool)
day_mask[6:19] = True

rows = []

for name, pred in predictions.items():
    mae, rmse, r2 = calculate(Y, pred)
    d_mae, d_rmse, d_r2 = calculate(
        Y[:, day_mask],
        pred[:, day_mask]
    )

    rows.append({
        "model": name,
        "mae_mw": mae,
        "rmse_mw": rmse,
        "r2": r2,
        "day_mae_mw": d_mae,
        "day_rmse_mw": d_rmse,
        "day_r2": d_r2
    })

results = pd.DataFrame(rows)
results.to_csv(
    OUT / "daily_model_comparison.csv",
    index=False
)

print("\n=== DAILY FORECAST COMPARISON ===")
print(results.round(4).to_string(index=False))

# Save one complete daily example for plotting.
example = pd.DataFrame({
    "Timestamp": pd.date_range(
        forecast_times.iloc[0], periods=24, freq="h"
    ),
    "Actual_MW": Y[0],
    "LSTM_MW": predictions["LSTM"][0],
    "RandomForest_MW": predictions["Random Forest"][0],
    "Persistence24h_MW": predictions["Persistence 24h"][0]
})

example.to_csv(
    OUT / "daily_forecast_example.csv",
    index=False
)

print("\nSaved daily comparison and example forecast.")
