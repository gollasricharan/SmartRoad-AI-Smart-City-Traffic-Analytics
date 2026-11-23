
import pandas as pd

# Load datasets
city = pd.read_csv("data/raw/hyderabad_accidents.csv")
state = pd.read_csv("data/raw/telangana_accidents.csv")
weather = pd.read_csv("data/raw/hyderabad_weather.csv")

print("\n===== HYDERABAD ACCIDENT DATA =====")
print(city)
print("Records:", len(city))

print("\n===== TELANGANA ACCIDENT DATA =====")
print(state)
print("Records:", len(state))

print("\n===== HYDERABAD WEATHER DATA =====")
print(weather.head())
print("Records:", len(weather))

# Check missing values
print("\n===== MISSING VALUES =====")

print("\nHyderabad:")
print(city.isnull().sum())

print("\nTelangana:")
print(state.isnull().sum())

print("\nWeather:")
print(weather.isnull().sum())

# Check duplicate records
print("\n===== DUPLICATE RECORDS =====")

print("Hyderabad:", city.duplicated().sum())
print("Telangana:", state.duplicated().sum())
print("Weather:", weather.duplicated().sum())

print("\nDataset verification completed!")
