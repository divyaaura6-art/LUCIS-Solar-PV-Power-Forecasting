
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE / "processed" / "india_pv_hourly.csv",
    parse_dates=["Timestamp"]
)

df = df.sort_values("Timestamp").reset_index(drop=True)

# Longest uninterrupted segment
df = df[
    df["Timestamp"] >= pd.Timestamp("2022-12-06")
].copy()

assert df["PV_Power_MW"].notna().all(), (
    "Missing PV values found in selected segment"
)

# NeuralForecast input format
df = df.rename(
    columns={
        "Timestamp": "ds",
        "PV_Power_MW": "y"
    }
)

df["unique_id"] = "SolarPlant_1"

df["y"] = df["y"] / 5.0

df = df[["unique_id", "ds", "y"]]

train = df[
    df["ds"] < "2025-01-01"
].copy()

val = df[
    (df["ds"] >= "2025-01-01") &
    (df["ds"] < "2026-01-01")
].copy()

test = df[
    df["ds"] >= "2026-01-01"
].copy()

out = BASE / "processed" / "nhits_data"
out.mkdir(parents=True, exist_ok=True)

train.to_csv(out / "train.csv", index=False)
val.to_csv(out / "val.csv", index=False)
test.to_csv(out / "test.csv", index=False)

print("Training hours:", len(train))
print("Validation hours:", len(val))
print("Test hours:", len(test))

print("\nTrain:", train["ds"].min(), "to", train["ds"].max())
print("Validation:", val["ds"].min(), "to", val["ds"].max())
print("Test:", test["ds"].min(), "to", test["ds"].max())

print("\nN-HiTS dataset prepared successfully!")
