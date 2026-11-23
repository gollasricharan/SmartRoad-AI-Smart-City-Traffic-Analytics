
import requests
import pandas as pd
import os

# Hyderabad coordinates
latitude = 17.3850
longitude = 78.4867

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2019-01-01",
    "end_date": "2023-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "weather_code"
    ],
    "timezone": "Asia/Kolkata"
}

print("Downloading Hyderabad weather data...")

response = requests.get(url, params=params, timeout=120)
response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["hourly"])

os.makedirs("data/raw", exist_ok=True)

df.to_csv("data/raw/hyderabad_weather.csv", index=False)

print("Weather data downloaded successfully!")
print("Total records:", len(df))
print(df.head())
