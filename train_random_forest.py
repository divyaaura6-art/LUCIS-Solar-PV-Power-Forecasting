
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"
MODELS = BASE / "models"

OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

CAPACITY = 5.0

print("Loading datasets...")

X_train = np.load(DATA / "X_train.npy").reshape(-1, 168)
y_train = np.load(DATA / "y_train.npy")

X_val = np.load(DATA / "X_val.npy").reshape(-1, 168)
y_val = np.load(DATA / "y_val.npy")

X_test = np.load(DATA / "X_test.npy").reshape(-1, 168)
y_test = np.load(DATA / "y_test.npy")

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# Restrict parallel processing to reduce laptop load.
model = RandomForestRegressor(
    n_estimators=60,
    max_depth=16,
    min_samples_leaf=3,
    max_features=0.7,
    n_jobs=2,
    random_state=42
)

print("\nTraining Random Forest...")
start = time.time()

model.fit(X_train, y_train)

print(f"Training completed in {time.time()-start:.1f} seconds")

def metrics(actual, predicted):
    return {
        "mae": mean_absolute_error(actual.ravel(), predicted.ravel()),
        "rmse": np.sqrt(mean_squared_error(
            actual.ravel(), predicted.ravel()
        )),
        "r2": r2_score(actual.ravel(), predicted.ravel())
    }

# Validation performance
val_pred = np.clip(model.predict(X_val), 0, 1)
val_results = metrics(
    y_val * CAPACITY,
    val_pred * CAPACITY
)

print("\n=== VALIDATION RESULTS ===")
print(f"MAE : {val_results['mae']:.4f} MW")
print(f"RMSE: {val_results['rmse']:.4f} MW")
print(f"R2  : {val_results['r2']:.4f}")

# Held-out test performance
test_pred = np.clip(model.predict(X_test), 0, 1)

actual = y_test * CAPACITY
predicted = test_pred * CAPACITY

results = metrics(actual, predicted)

hours = (
    timestamps["forecast_start"].dt.hour.to_numpy()[:, None]
    + np.arange(24)
) % 24

daylight = (hours >= 6) & (hours < 19)

day_results = metrics(
    actual[daylight],
    predicted[daylight]
)

print("\n=== RANDOM FOREST TEST RESULTS ===")
print(f"Overall MAE : {results['mae']:.4f} MW")
print(f"Overall RMSE: {results['rmse']:.4f} MW")
print(f"Overall R2  : {results['r2']:.4f}")

print("\n=== DAYLIGHT-ONLY RESULTS ===")
print(f"Day MAE     : {day_results['mae']:.4f} MW")
print(f"Day RMSE    : {day_results['rmse']:.4f} MW")
print(f"Day R2      : {day_results['r2']:.4f}")

pd.DataFrame([{
    "model": "Random Forest",
    "mae_mw": results["mae"],
    "rmse_mw": results["rmse"],
    "r2": results["r2"],
    "day_mae_mw": day_results["mae"],
    "day_rmse_mw": day_results["rmse"],
    "day_r2": day_results["r2"]
}]).to_csv(
    OUT / "random_forest_test_metrics.csv",
    index=False
)

joblib.dump(
    model,
    MODELS / "random_forest.joblib",
    compress=3
)

print("\nModel and metrics saved successfully!")
