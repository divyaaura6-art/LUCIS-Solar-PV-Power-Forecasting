
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

df = pd.read_csv(
    OUT / "nhits_daily_predictions.csv",
    parse_dates=["Timestamp"]
)

# Choose a complete day from the test period
selected_date = pd.Timestamp("2026-06-15")

day = df[
    df["Timestamp"].dt.normalize() == selected_date
].sort_values("Timestamp")

if len(day) != 24:
    raise ValueError(
        f"Expected 24 hourly observations, found {len(day)}"
    )

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    day["Timestamp"],
    day["Actual_MW"],
    marker="o",
    linewidth=2,
    label="Actual Solar Power"
)

ax.plot(
    day["Timestamp"],
    day["NHITS_MW"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="N-HiTS Prediction"
)

ax.set_title(
    "Actual vs Predicted Solar Power — N-HiTS\n"
    "15 June 2026"
)
ax.set_xlabel("Time")
ax.set_ylabel("Solar Power (MW)")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()

fig.autofmt_xdate()
plt.tight_layout()

filename = OUT / "nhits_actual_vs_predicted.png"

plt.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved:", filename)
print("\nHourly actual and predicted values:")
print(
    day[
        ["Timestamp", "Actual_MW", "NHITS_MW"]
    ].to_string(index=False)
)
