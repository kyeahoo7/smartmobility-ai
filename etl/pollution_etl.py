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

# Coordonnées Paris
LATITUDE = 48.8566
LONGITUDE = 2.3522

URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&hourly=pm10,pm2_5,nitrogen_dioxide,european_aqi"
    "&timezone=Europe%2FParis"
)

print("Téléchargement des données pollution...")

response = requests.get(URL)

if response.status_code != 200:
    raise Exception(f"Erreur API Pollution : {response.status_code}")

data = response.json()

hourly = data["hourly"]

df = pd.DataFrame({
    "timestamp": hourly["time"],
    "pm10": hourly["pm10"],
    "pm25": hourly["pm2_5"],
    "no2": hourly["nitrogen_dioxide"],
    "aqi": hourly["european_aqi"]
})

df["timestamp"] = pd.to_datetime(df["timestamp"])

# MVP : location_id = 1
df["location_id"] = 1

# Fonction classification AQI
def classify_aqi(aqi):
    if aqi <= 20:
        return "Très bon"
    elif aqi <= 40:
        return "Bon"
    elif aqi <= 60:
        return "Moyen"
    elif aqi <= 80:
        return "Dégradé"
    else:
        return "Mauvais"

df["air_quality_index"] = df["aqi"].apply(classify_aqi)

# Colonnes finales
df = df[[
    "location_id",
    "timestamp",
    "no2",
    "pm10",
    "pm25",
    "air_quality_index"
]]

# Backup CSV
df.to_csv("data/raw/pollution_data.csv", index=False)

# Insertion PostgreSQL
df.to_sql(
    "pollution_data",
    engine,
    if_exists="append",
    index=False
)

print("Données pollution insérées avec succès")
print(df.head())