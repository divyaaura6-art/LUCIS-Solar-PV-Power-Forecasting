
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / "processed" / "india_pv_hourly.csv"
OUT = BASE / "processed" / "lstm_data"
OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK = 168
HORIZON = 24
CAPACITY_MW = 5.0

df = pd.read_csv(INPUT, parse_dates=["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Verify the complete hourly timeline.
expected = pd.date_range(
    df["Timestamp"].iloc[0],
    df["Timestamp"].iloc[-1],
    freq="h"
)

assert len(df) == len(expected), "Unexpected timeline length"
assert df["Timestamp"].equals(pd.Series(expected)), \
    "Hourly timestamps contain gaps or duplicates"

power = df["PV_Power_MW"].to_numpy(dtype=np.float32)

valid = (
    (df["Valid_15min_Readings"].to_numpy() == 4)
    & np.isfinite(power)
    & (power >= 0)
    & (power <= CAPACITY_MW)
)

# Normalize by the known plant capacity.
# No scaler is fitted on future/test observations.
normalized = power / CAPACITY_MW

# Chronological boundaries (based on target time).
train_end = pd.Timestamp("2024-12-31 23:00:00")
val_end = pd.Timestamp("2025-12-31 23:00:00")

X = {"train": [], "val": [], "test": []}
Y = {"train": [], "val": [], "test": []}
T = {"train": [], "val": [], "test": []}

# Prefix sum lets us check missing data in O(1).
invalid_prefix = np.concatenate([
    [0], np.cumsum(~valid)
])

for start in range(LOOKBACK, len(df) - HORIZON + 1):
    left = start - LOOKBACK
    right = start + HORIZON

    # Reject windows containing incomplete readings.
    if invalid_prefix[right] != invalid_prefix[left]:
        continue

    target_start = df["Timestamp"].iloc[start]
    target_end = df["Timestamp"].iloc[right - 1]

    # Keep each 24-hour target entirely inside one split.
    if target_end <= train_end:
        split = "train"
    elif target_start > train_end and target_end <= val_end:
        split = "val"
    elif target_start > val_end:
        split = "test"
    else:
        continue

    X[split].append(normalized[left:start])
    Y[split].append(normalized[start:right])
    T[split].append(target_start.isoformat())

for split in ("train", "val", "test"):
    x = np.asarray(X[split], dtype=np.float32)
    y = np.asarray(Y[split], dtype=np.float32)

    if len(x) == 0:
        raise RuntimeError(f"No valid {split} sequences")

    x = x.reshape(-1, LOOKBACK, 1)

    np.save(OUT / f"X_{split}.npy", x)
    np.save(OUT / f"y_{split}.npy", y)

    pd.DataFrame({
        "forecast_start": T[split]
    }).to_csv(
        OUT / f"timestamps_{split}.csv",
        index=False
    )

    print(
        f"{split.upper():5s}: "
        f"{len(x):6d} sequences | "
        f"X {x.shape} | y {y.shape}"
    )

print()
print("Preparation complete!")
print("Saved to:", OUT)
