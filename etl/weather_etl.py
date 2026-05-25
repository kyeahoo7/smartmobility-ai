import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Coordonnées de Paris
LATITUDE = 48.8566
LONGITUDE = 2.3522

URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
    "&timezone=Europe%2FParis"
    "&forecast_days=2"
)

print("Téléchargement des données météo...")

response = requests.get(URL)

if response.status_code != 200:
    raise Exception(f"Erreur API Open-Meteo : {response.status_code}")

data = response.json()
hourly = data["hourly"]

df = pd.DataFrame({
    "timestamp": hourly["time"],
    "temperature": hourly["temperature_2m"],
    "humidity": hourly["relative_humidity_2m"],
    "wind_speed": hourly["wind_speed_10m"],
    "precipitation": hourly["precipitation"]
})

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Paris = location_id 1 par défaut MVP
# Si tu as déjà plusieurs locations, on associe météo globale Paris à toutes les zones plus tard.
df["location_id"] = 1

df = df[[
    "location_id",
    "timestamp",
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation"
]]

df.to_csv("data/raw/weather_data.csv", index=False)

df.to_sql(
    "weather_data",
    engine,
    if_exists="append",
    index=False
)

print("Données météo insérées avec succès")
print(df.head())