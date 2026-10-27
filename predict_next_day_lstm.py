
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Load our selected LSTM
model = load_model(
    BASE / "models" / "lstm_best.keras",
    compile=False
)

# Load the latest available observations
df = pd.read_csv(
    BASE / "processed" / "nhits_data" / "test.csv",
    parse_dates=["ds"]
)

history = df.tail(168).copy()

assert len(history) == 168
assert history["y"].notna().all()
assert history["ds"].diff().iloc[1:].eq(
    pd.Timedelta(hours=1)
).all()

# Data is already normalized by 5 MW
X = history["y"].to_numpy(
    dtype=np.float32
).reshape(1, 168, 1)

# Predict 24 hours
prediction = model.predict(X, verbose=0)[0]

# Convert normalized values to MW
prediction_mw = np.clip(
    prediction * 5.0,
    0.0,
    5.0
)

forecast_start = (
    history["ds"].max() + pd.Timedelta(hours=1)
)

timestamps = pd.date_range(
    start=forecast_start,
    periods=24,
    freq="h"
)

result = pd.DataFrame({
    "Timestamp": timestamps,
    "Predicted_MW": prediction_mw
})

result.to_csv(
    OUT / "lstm_next_day_forecast.csv",
    index=False
)

plt.figure(figsize=(11, 5))

plt.plot(
    result["Timestamp"],
    result["Predicted_MW"],
    marker="o",
    linewidth=2,
    label="Original LSTM Forecast"
)

plt.title("Next-Day Solar Power Forecast — Original LSTM")
plt.xlabel("Time")
plt.ylabel("Predicted Solar Power (MW)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "lstm_next_day_forecast.png",
    dpi=300
)

plt.show()

print("\nORIGINAL LSTM NEXT-DAY FORECAST")
print(result.to_string(index=False))
print("\nForecast saved successfully!")
