
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "processed" / "lstm_data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Load the existing test data
X_test = np.load(DATA / "X_test.npy")
y_test = np.load(DATA / "y_test.npy")

timestamps = pd.read_csv(
    DATA / "timestamps_test.csv",
    parse_dates=["forecast_start"]
)

# Select the same midnight-issued forecasts
mask = (
    (timestamps["forecast_start"].dt.hour == 0) &
    (timestamps["forecast_start"].dt.minute == 0)
)

indices = np.where(mask.to_numpy())[0]

assert len(indices) == 243
assert len(X_test) == len(y_test) == len(timestamps)

X_daily = X_test[indices]
y_daily = y_test[indices]

forecast_starts = timestamps.iloc[indices][
    "forecast_start"
].reset_index(drop=True)

actual_mw = y_daily * 5.0

models = {
    "Attention_LSTM": "attention_lstm_best.keras",
    "CNN_LSTM": "cnn_lstm_best.keras"
}

summary = []

for name, filename in models.items():

    print(f"\nEvaluating {name}...")

    model = load_model(
        BASE / "models" / filename,
        compile=False
    )

    assert model.input_shape[1:] == (168, 1)
    assert model.output_shape[-1] == 24

    predictions = model.predict(
        X_daily,
        batch_size=64,
        verbose=0
    )

    predicted_mw = np.clip(
        predictions * 5.0,
        0.0,
        5.0
    )

    records = []

    for i, start in enumerate(forecast_starts):
        for hour in range(24):
            records.append({
                "Timestamp": start + pd.Timedelta(hours=hour),
                "Actual_MW": float(actual_mw[i, hour]),
                "Predicted_MW": float(predicted_mw[i, hour])
            })

    result = pd.DataFrame(records)

    assert len(result) == 5832
    assert result["Actual_MW"].notna().all()
    assert result["Predicted_MW"].notna().all()

    result.to_csv(
        OUT / f"{name.lower()}_daily_predictions.csv",
        index=False
    )

    # Full-day evaluation
    mae = mean_absolute_error(
        result["Actual_MW"],
        result["Predicted_MW"]
    )

    rmse = np.sqrt(mean_squared_error(
        result["Actual_MW"],
        result["Predicted_MW"]
    ))

    r2 = r2_score(
        result["Actual_MW"],
        result["Predicted_MW"]
    )

    # Daytime evaluation: 06:00–18:00
    daytime = result[
        result["Timestamp"].dt.hour.between(6, 18)
    ]

    day_mae = mean_absolute_error(
        daytime["Actual_MW"],
        daytime["Predicted_MW"]
    )

    day_rmse = np.sqrt(mean_squared_error(
        daytime["Actual_MW"],
        daytime["Predicted_MW"]
    ))

    day_r2 = r2_score(
        daytime["Actual_MW"],
        daytime["Predicted_MW"]
    )

    summary.append({
        "Model": name,
        "MAE_MW": mae,
        "RMSE_MW": rmse,
        "R2": r2,
        "Daytime_MAE_MW": day_mae,
        "Daytime_RMSE_MW": day_rmse,
        "Daytime_R2": day_r2
    })

    print(f"MAE: {mae:.4f} MW")
    print(f"RMSE: {rmse:.4f} MW")
    print(f"R2: {r2:.4f}")
    print(f"Daytime MAE: {day_mae:.4f} MW")
    print(f"Daytime RMSE: {day_rmse:.4f} MW")
    print(f"Daytime R2: {day_r2:.4f}")

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "remaining_lstm_comparison.csv",
    index=False
)

print("\nFINAL RESULTS")
print(summary_df.to_string(index=False))
print("\nEvaluation completed successfully!")
