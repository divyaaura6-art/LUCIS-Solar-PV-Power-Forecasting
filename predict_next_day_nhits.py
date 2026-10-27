
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Load trained model
nf = NeuralForecast.load(
    path=str(BASE / "models" / "nhits")
)

# Load latest observed data
df = pd.read_csv(
    BASE / "processed" / "nhits_data" / "test.csv",
    parse_dates=["ds"]
)

# Last 7 days = 168 hours
history = df.tail(168).copy()

assert len(history) == 168
assert history["y"].notna().all()
assert history["ds"].diff().iloc[1:].eq(
    pd.Timedelta(hours=1)
).all()

# Predict next 24 hours
forecast = nf.predict(df=history)

model_col = [
    c for c in forecast.columns
    if c not in ["unique_id", "ds"]
][0]

forecast = forecast.sort_values("ds")

# Convert normalized output to MW
forecast["Predicted_MW"] = np.clip(
    forecast[model_col].to_numpy() * 5.0,
    0.0,
    5.0
)

result = forecast[["ds", "Predicted_MW"]].copy()

# Save predictions
result.to_csv(
    OUT / "nhits_next_day_forecast.csv",
    index=False
)

# Plot
plt.figure(figsize=(11, 5))
plt.plot(
    result["ds"],
    result["Predicted_MW"],
    marker="o",
    linewidth=2,
    label="N-HiTS Forecast"
)

plt.title("Next-Day Solar Power Forecast — N-HiTS")
plt.xlabel("Time")
plt.ylabel("Predicted Power (MW)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "nhits_next_day_forecast.png",
    dpi=300
)

plt.show()

print("\nNEXT-DAY SOLAR FORECAST")
print(result.to_string(index=False))
print("\nForecast saved successfully!")
