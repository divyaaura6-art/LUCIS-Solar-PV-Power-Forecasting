
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

CAPACITY = 5.0

X = np.load(DATA / "X_test.npy")[:, :, 0] * CAPACITY
Y = np.load(DATA / "y_test.npy") * CAPACITY

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

# Previous day's 24-hour profile.
persistence_24h = X[:, -24:]

# Same 24-hour period from seven days earlier.
persistence_7d = X[:, :24]

def evaluate(name, prediction):
    mae = mean_absolute_error(Y.ravel(), prediction.ravel())
    rmse = np.sqrt(mean_squared_error(Y.ravel(), prediction.ravel()))
    r2 = r2_score(Y.ravel(), prediction.ravel())

    hours = (
        timestamps["forecast_start"].dt.hour.to_numpy()[:, None]
        + np.arange(24)
    ) % 24

    daylight = (hours >= 6) & (hours < 19)

    day_mae = mean_absolute_error(
        Y[daylight], prediction[daylight]
    )
    day_rmse = np.sqrt(mean_squared_error(
        Y[daylight], prediction[daylight]
    ))
    day_r2 = r2_score(
        Y[daylight], prediction[daylight]
    )

    print(f"\n{name}")
    print(f"Overall MAE : {mae:.4f} MW")
    print(f"Overall RMSE: {rmse:.4f} MW")
    print(f"Overall R2  : {r2:.4f}")
    print(f"Day MAE     : {day_mae:.4f} MW")
    print(f"Day RMSE    : {day_rmse:.4f} MW")
    print(f"Day R2      : {day_r2:.4f}")

    return {
        "model": name,
        "mae_mw": mae,
        "rmse_mw": rmse,
        "r2": r2,
        "day_mae_mw": day_mae,
        "day_rmse_mw": day_rmse,
        "day_r2": day_r2
    }

results = [
    evaluate("Persistence_24h", persistence_24h),
    evaluate("Persistence_7d", persistence_7d)
]

pd.DataFrame(results).to_csv(
    OUT / "baseline_test_metrics.csv",
    index=False
)

print("\nSaved baseline metrics.")
