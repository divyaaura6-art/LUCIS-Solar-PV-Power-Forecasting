
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE / "processed" / "india_pv_hourly.csv",
    parse_dates=["Timestamp"]
)

df = df.sort_values("Timestamp").reset_index(drop=True)

# Missing hours break a continuous sequence
valid = df["PV_Power_MW"].notna()

# Assign a separate segment whenever a missing hour occurs
segment_id = (~valid).cumsum()

df = df[valid].copy()
df["segment_id"] = segment_id[valid].values

segments = (
    df.groupby("segment_id")
    .agg(
        start=("Timestamp", "min"),
        end=("Timestamp", "max"),
        hours=("Timestamp", "size")
    )
    .reset_index()
)

# N-HiTS requires enough history and future observations
segments = segments[segments["hours"] >= 192]

print("Usable continuous segments:", len(segments))
print("Total usable hours:", segments["hours"].sum())
print("\nLargest segments:")
print(
    segments.sort_values("hours", ascending=False)
    .head(10)
    .to_string(index=False)
)

segments.to_csv(
    BASE / "processed" / "nhits_segments.csv",
    index=False
)

print("\nSaved: processed/nhits_segments.csv")
