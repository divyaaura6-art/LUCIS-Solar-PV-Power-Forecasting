
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROCESSED = BASE / "processed"

pv = pd.read_csv(
    PROCESSED / "india_pv_hourly.csv",
    parse_dates=["Timestamp"]
)

weather = pd.read_csv(
    PROCESSED / "india_weather_hourly_preliminary.csv"
)

# Interpret PV timestamps as local IST interval starts.
pv["Timestamp"] = pd.to_datetime(pv["Timestamp"])

# ERA5-Land timestamps are already converted to IST.
weather["timestamp_ist"] = pd.to_datetime(
    weather["timestamp_ist"],
    utc=True
).dt.tz_convert("Asia/Kolkata")

# Remove timezone information while preserving IST clock time.
weather["timestamp_ist"] = (
    weather["timestamp_ist"].dt.tz_localize(None)
)

weather = weather.sort_values("timestamp_ist")

# Each ERA5-Land irradiation value represents the
# preceding one-hour interval ending at timestamp_ist.
# Estimate irradiation over IST clock-hour intervals
# by weighting two overlapping ERA5-Land intervals.
weather["solar_irradiation_next"] = (
    weather["solar_irradiation_whm2"].shift(-1)
)

weather["solar_irradiation_ist_hour"] = (
    0.5 * weather["solar_irradiation_whm2"]
    + 0.5 * weather["solar_irradiation_next"]
)

# The ERA5 timestamps are at XX:30 IST.
# Their irradiation intervals are XX-1:30 to XX:30.
# Combining the interval ending XX:30 with the next
# gives the IST interval XX:00 to XX+1:00.
weather["Timestamp"] = (
    weather["timestamp_ist"].dt.floor("h")
)

# Temperature and wind at XX:30 are used as
# representative midpoint observations for the IST hour.
features = weather[[
    "Timestamp",
    "temperature_c",
    "wind_speed_ms",
    "solar_irradiation_ist_hour"
]]

merged = pv.merge(
    features,
    on="Timestamp",
    how="inner",
    validate="one_to_one"
)

# Keep only hours with all four 15-minute PV readings.
merged["pv_complete"] = (
    merged["Valid_15min_Readings"] == 4
)

merged.loc[
    ~merged["pv_complete"], "PV_Power_MW"
] = np.nan

merged = merged.sort_values("Timestamp")

output = PROCESSED / "india_merged_preliminary.csv"
merged.to_csv(output, index=False)

print("Merge completed!")
print("Merged rows:", len(merged))
print("Date range:",
      merged["Timestamp"].min(),
      "to",
      merged["Timestamp"].max())
print("Missing values:")
print(merged.isna().sum())
print("Saved to:", output)
