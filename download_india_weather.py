
from pathlib import Path
import cdsapi

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "weather"
OUTPUT_DIR.mkdir(exist_ok=True)

client = cdsapi.Client()

DATASET = "reanalysis-era5-land"

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

# Approximate Addakal reference point.
# A small surrounding area allows us to inspect nearby grid cells.
AREA = [16.8, 77.8, 16.6, 78.0]

for year in range(2019, 2027):
    if year == 2019:
        months = range(3, 13)
    elif year == 2026:
        months = range(1, 9)
    else:
        months = range(1, 13)

    for month in months:
        target = OUTPUT_DIR / f"era5_india_{year}_{month:02d}.nc"

        if target.exists() and target.stat().st_size > 0:
            print(f"Skipping existing file: {target.name}")
            continue

        request = {
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        print(f"Downloading {year}-{month:02d}...")

        try:
            client.retrieve(DATASET, request, str(target))
            print(f"Saved: {target.name}")
        except Exception as error:
            print(f"FAILED {year}-{month:02d}: {error}")
            print("You can rerun the script later.")

print("Weather download requests finished.")
