
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(__file__).resolve().parent
WEATHER_DIR = BASE / "weather"
OUTPUT_DIR = BASE / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# Provisional Addakal reference location.
# Exact plant coordinates have not been verified.
LAT = 16.6
LON = 77.8

files = sorted(WEATHER_DIR.glob("era5_india_????_??.nc"))

if not files:
    raise FileNotFoundError("No monthly ERA5-Land files found.")

print(f"Found {len(files)} monthly weather files.")

frames = []

for file in files:
    print(f"Processing {file.name}")

    with xr.open_dataset(file) as ds:
        point = ds.sel(
            latitude=LAT,
            longitude=LON,
            method="nearest"
        )

        df = point[
            ["t2m", "u10", "v10", "ssrd"]
        ].to_dataframe().reset_index()

        df = df[[
            "valid_time", "t2m", "u10", "v10", "ssrd"
        ]].copy()

        frames.append(df)

weather = pd.concat(frames, ignore_index=True)

# Remove duplicate timestamps if any.
weather = weather.drop_duplicates(subset=["valid_time"])
weather = weather.sort_values("valid_time")

# Convert UTC timestamps to IST.
weather["timestamp_ist"] = (
    pd.to_datetime(weather["valid_time"], utc=True)
    .dt.tz_convert("Asia/Kolkata")
)

# Temperature: Kelvin -> Celsius.
weather["temperature_c"] = weather["t2m"] - 273.15

# Wind speed from U and V components.
weather["wind_speed_ms"] = np.sqrt(
    weather["u10"] ** 2 + weather["v10"] ** 2
)

# ERA5-Land hourly accumulated radiation (J/m2)
# -> hourly irradiation (Wh/m2).
# The accumulation convention must be verified before
# interpreting these values as hourly radiation.

# ERA5-Land ssrd is accumulated from the start of
# the forecast day. Convert it to hourly irradiation.

weather["utc_hour"] = pd.to_datetime(
    weather["valid_time"]
).dt.hour

# Difference consecutive accumulated values.
radiation_diff = weather["ssrd"].diff()

# At 01:00 UTC, accumulation restarts for the new day.
# At 00:00 UTC, subtract the previous day's 23:00
# accumulation to obtain the final hour of that day.
weather["solar_irradiation_whm2"] = np.where(
    weather["utc_hour"] == 1,
    weather["ssrd"] / 3600.0,
    radiation_diff / 3600.0
)

# The first timestamp has no previous accumulation.
weather.loc[
    weather.index[0], "solar_irradiation_whm2"
] = np.nan

# Small negative values can occur from numerical precision.
weather["solar_irradiation_whm2"] = (
    weather["solar_irradiation_whm2"].clip(lower=0)
)


output = weather[[
    "valid_time",
    "timestamp_ist",
    "temperature_c",
    "wind_speed_ms",
    "solar_irradiation_whm2"
]]

target = OUTPUT_DIR / "india_weather_hourly_preliminary.csv"
output.to_csv(target, index=False)

print()
print("Weather preprocessing completed!")
print("Total hourly rows:", len(output))
print("Date range (UTC):",
      output["valid_time"].min(),
      "to",
      output["valid_time"].max())
print("Missing values:")
print(output.isna().sum())
print("Saved to:", target)
