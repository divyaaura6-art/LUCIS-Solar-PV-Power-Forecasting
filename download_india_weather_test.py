
from pathlib import Path
import cdsapi

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "weather"
OUTPUT_DIR.mkdir(exist_ok=True)

client = cdsapi.Client()

dataset = "reanalysis-era5-land"

request = {
    "variable": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_solar_radiation_downwards"
    ],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01", "02"],
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": [16.8, 77.8, 16.6, 78.0],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

target = OUTPUT_DIR / "india_weather_test.nc"

print("Starting ERA5-Land test download...")

client.retrieve(dataset, request, str(target))

print("Download completed!")
print("Saved to:", target)
