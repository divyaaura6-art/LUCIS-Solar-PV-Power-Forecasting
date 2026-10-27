
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

lstm = pd.read_csv(
    OUT / "lstm_daily_predictions.csv",
    parse_dates=["Timestamp"]
)

nhits = pd.read_csv(
    OUT / "nhits_daily_predictions.csv",
    parse_dates=["Timestamp"]
)

# Align predictions using identical timestamps
df = pd.merge(
    lstm[["Timestamp", "Actual_MW", "LSTM_MW"]],
    nhits[["Timestamp", "Actual_MW", "NHITS_MW"]],
    on="Timestamp",
    how="inner",
    suffixes=("_LSTM", "_NHITS"),
    validate="one_to_one"
)

assert len(df) == 243 * 24, (
    "Prediction timestamps are not fully aligned."
)

# Confirm both evaluations use identical actual values
difference = (
    df["Actual_MW_LSTM"] -
    df["Actual_MW_NHITS"]
).abs().max()

print("Maximum actual-value difference:", difference)

assert difference < 1e-4, (
    "Actual targets differ between models."
)

selected_date = pd.Timestamp("2026-06-15")

day = df[
    df["Timestamp"].dt.normalize() == selected_date
].sort_values("Timestamp")

assert len(day) == 24

plt.figure(figsize=(12, 6))

plt.plot(
    day["Timestamp"],
    day["Actual_MW_LSTM"],
    label="Actual Solar Power",
    marker="o",
    linewidth=2.5,
    color="black"
)

plt.plot(
    day["Timestamp"],
    day["LSTM_MW"],
    label="Original LSTM",
    marker="s",
    linestyle="--",
    linewidth=2
)

plt.plot(
    day["Timestamp"],
    day["NHITS_MW"],
    label="N-HiTS",
    marker="^",
    linestyle=":",
    linewidth=2
)

plt.title(
    "Actual vs Predicted Solar Power\n"
    "Original LSTM vs N-HiTS — 15 June 2026"
)

plt.xlabel("Time")
plt.ylabel("Solar Power (MW)")
plt.ylim(bottom=0)
plt.grid(alpha=0.3)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

filename = OUT / "final_actual_vs_predicted.png"

plt.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved:", filename)
print("Matched hourly observations:", len(df))
