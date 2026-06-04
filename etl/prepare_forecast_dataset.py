import os
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

print("Chargement des données trafic...")

traffic_df = pd.read_sql("""
    SELECT 
        t.id,
        t.location_id,
        t.timestamp,
        t.flow_rate,
        t.occupancy_rate,
        l.location_name
    FROM traffic_data t
    LEFT JOIN locations l ON t.location_id = l.id
""", engine)

traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
traffic_df["flow_rate"] = pd.to_numeric(traffic_df["flow_rate"], errors="coerce")
traffic_df["occupancy_rate"] = pd.to_numeric(traffic_df["occupancy_rate"], errors="coerce")

traffic_df = traffic_df.dropna(subset=[
    "location_name",
    "timestamp",
    "flow_rate",
    "occupancy_rate"
])

traffic_df = traffic_df.sort_values(["location_name", "timestamp"])

print("Nombre de lignes trafic :", len(traffic_df))
print("Nombre de capteurs :", traffic_df["location_name"].nunique())

print("Top capteurs par nombre de mesures :")
print(
    traffic_df["location_name"]
    .value_counts()
    .head(20)
)

# On garde seulement les capteurs avec au moins 3 mesures
sensor_counts = traffic_df["location_name"].value_counts()
valid_sensors = sensor_counts[sensor_counts >= 3].index

traffic_df = traffic_df[traffic_df["location_name"].isin(valid_sensors)]

# Features temporelles
traffic_df["hour"] = traffic_df["timestamp"].dt.hour
traffic_df["day_of_week"] = traffic_df["timestamp"].dt.dayofweek
traffic_df["month"] = traffic_df["timestamp"].dt.month
traffic_df["is_weekend"] = (traffic_df["day_of_week"] >= 5).astype(int)

# Encodage du capteur
traffic_df["sensor_code"] = traffic_df["location_name"].astype("category").cat.codes

# Variables retardées par nom de capteur
traffic_df["flow_rate_lag_1"] = traffic_df.groupby("location_name")["flow_rate"].shift(1)
traffic_df["occupancy_lag_1"] = traffic_df.groupby("location_name")["occupancy_rate"].shift(1)

# Cible : prochaine mesure du même capteur
traffic_df["flow_rate_next"] = traffic_df.groupby("location_name")["flow_rate"].shift(-1)

forecast_df = traffic_df.dropna(subset=[
    "flow_rate_lag_1",
    "occupancy_lag_1",
    "flow_rate_next"
])

final_columns = [
    "sensor_code",
    "flow_rate",
    "occupancy_rate",
    "flow_rate_lag_1",
    "occupancy_lag_1",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "flow_rate_next"
]

forecast_df = forecast_df[final_columns]

forecast_df.to_csv("data/processed/forecast_dataset.csv", index=False)

print("Dataset forecast créé avec succès")
print(forecast_df.head())
print("Shape :", forecast_df.shape)