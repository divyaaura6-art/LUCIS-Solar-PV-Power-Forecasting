
from pathlib import Path
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "nhits_data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

H = 24
SCALE = 5.0

print("Loading trained N-HiTS...")
nf = NeuralForecast.load(path=str(BASE / "models" / "nhits"))

train = pd.read_csv(DATA / "train.csv", parse_dates=["ds"])
val = pd.read_csv(DATA / "val.csv", parse_dates=["ds"])
test = pd.read_csv(DATA / "test.csv", parse_dates=["ds"])

# The model receives only observations available before
# each forecast origin. No future test targets are supplied.
history = pd.concat([train, val], ignore_index=True)
history = history.sort_values("ds").reset_index(drop=True)

predictions = []
daily_metrics = []

print("Evaluating daily midnight forecasts...")

# Forecast every test day independently
for day_start in pd.date_range(
    test["ds"].min().normalize(),
    test["ds"].max().normalize(),
    freq="D"
):
    day_end = day_start + pd.Timedelta(hours=23)

    actual = test[
        (test["ds"] >= day_start) &
        (test["ds"] <= day_end)
    ].sort_values("ds")

    if len(actual) != H:
        continue

    # Use only the latest 168 observed hours
    available = history[
        history["ds"] < day_start
    ].tail(168)

    if len(available) != 168:
        continue

    expected = pd.date_range(
        end=day_start - pd.Timedelta(hours=1),
        periods=168,
        freq="h"
    )

    if not np.array_equal(
        available["ds"].to_numpy(),
        expected.to_numpy()
    ):
        continue

    # Forecast the next 24 hours
    forecast = nf.predict(df=available)

    model_col = [
        c for c in forecast.columns
        if c not in ["unique_id", "ds"]
    ][0]

    forecast = forecast.sort_values("ds")

    if not np.array_equal(
        forecast["ds"].to_numpy(),
        actual["ds"].to_numpy()
    ):
        raise ValueError(
            f"Forecast timestamp mismatch: {day_start}"
        )

    y_true = actual["y"].to_numpy() * SCALE
    y_pred = forecast[model_col].to_numpy() * SCALE

    # Physical power limits of our 5 MW plant
    y_pred = np.clip(y_pred, 0.0, 5.0)

    for timestamp, truth, pred in zip(
        actual["ds"], y_true, y_pred
    ):
        predictions.append({
            "Timestamp": timestamp,
            "Actual_MW": truth,
            "NHITS_MW": pred
        })

    daily_metrics.append({
        "Date": day_start,
        "MAE_MW": mean_absolute_error(y_true, y_pred),
        "RMSE_MW": np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    })

    # Update history only AFTER the forecast is made.
    # This simulates daily forecasting with newly observed
    # generation data available on the following day.
    history = pd.concat(
        [history, actual],
        ignore_index=True
    )

    if len(daily_metrics) % 30 == 0:
        print("Completed", len(daily_metrics), "days")

if not predictions:
    raise RuntimeError("No valid forecasts were generated.")

pred_df = pd.DataFrame(predictions)
pred_df.to_csv(
    OUT / "nhits_daily_predictions.csv",
    index=False
)

pd.DataFrame(daily_metrics).to_csv(
    OUT / "nhits_per_day_metrics.csv",
    index=False
)

y_true = pred_df["Actual_MW"].to_numpy()
y_pred = pred_df["NHITS_MW"].to_numpy()

daylight = pred_df["Timestamp"].dt.hour.between(6, 18)

def metrics(actual, predicted):
    return {
        "MAE_MW": mean_absolute_error(actual, predicted),
        "RMSE_MW": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted)
    }

overall = metrics(y_true, y_pred)
day = metrics(
    pred_df.loc[daylight, "Actual_MW"],
    pred_df.loc[daylight, "NHITS_MW"]
)

result = {
    "Model": "N-HiTS",
    "Days": len(daily_metrics),
    **overall,
    "Daylight_MAE_MW": day["MAE_MW"],
    "Daylight_RMSE_MW": day["RMSE_MW"],
    "Daylight_R2": day["R2"]
}

pd.DataFrame([result]).to_csv(
    OUT / "nhits_daily_metrics.csv",
    index=False
)

print("\n========== N-HiTS RESULTS ==========")
print("Forecast days:", result["Days"])
print("MAE:", round(result["MAE_MW"], 4), "MW")
print("RMSE:", round(result["RMSE_MW"], 4), "MW")
print("R2:", round(result["R2"], 4))
print("Daylight MAE:", round(result["Daylight_MAE_MW"], 4))
print("Daylight RMSE:", round(result["Daylight_RMSE_MW"], 4))
print("Daylight R2:", round(result["Daylight_R2"], 4))
print("====================================")
