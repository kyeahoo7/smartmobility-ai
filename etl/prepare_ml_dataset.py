# import os
# import pandas as pd
# from dotenv import load_dotenv
# from sqlalchemy import create_engine

# load_dotenv()

# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")

# engine = create_engine(
#     f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )

# print("Chargement des données...")

# traffic_df = pd.read_sql(
#     "SELECT * FROM traffic_data",
#     engine
# )

# weather_df = pd.read_sql(
#     "SELECT * FROM weather_data",
#     engine
# )

# pollution_df = pd.read_sql(
#     "SELECT * FROM pollution_data",
#     engine
# )

# # Conversion timestamps
# traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
# weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
# pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"])

# # Arrondi heure
# traffic_df["timestamp_hour"] = traffic_df["timestamp"].dt.floor("h")
# weather_df["timestamp_hour"] = weather_df["timestamp"].dt.floor("h")
# pollution_df["timestamp_hour"] = pollution_df["timestamp"].dt.floor("h")

# print("Fusion trafic + météo...")

# merged_df = traffic_df.merge(
#     weather_df,
#     on="timestamp_hour",
#     how="left",
#     suffixes=("", "_weather")
# )

# print("Fusion pollution...")

# merged_df = merged_df.merge(
#     pollution_df,
#     on="timestamp_hour",
#     how="left",
#     suffixes=("", "_pollution")
# )

# # Features temporelles
# merged_df["hour"] = merged_df["timestamp_hour"].dt.hour
# merged_df["day_of_week"] = merged_df["timestamp_hour"].dt.dayofweek
# merged_df["month"] = merged_df["timestamp_hour"].dt.month
# merged_df["is_weekend"] = (
#     merged_df["day_of_week"] >= 5
# ).astype(int)

# # Création automatique de la cible IA à partir du débit
# merged_df["flow_rate"] = pd.to_numeric(
#     merged_df["flow_rate"],
#     errors="coerce"
# )

# merged_df["occupancy_rate"] = pd.to_numeric(
#     merged_df["occupancy_rate"],
#     errors="coerce"
# )

# merged_df = merged_df.dropna(subset=["flow_rate"])

# def classify_congestion(flow):
#     if flow < 300:
#         return 0  # faible
#     elif flow < 800:
#         return 1  # moyenne
#     else:
#         return 2  # forte

# merged_df["congestion_target"] = merged_df["flow_rate"].apply(classify_congestion)

# # Nettoyage
# # Remplissage des valeurs manquantes météo/pollution
# numeric_columns = [
#     "temperature",
#     "humidity",
#     "wind_speed",
#     "precipitation",
#     "no2",
#     "pm10",
#     "pm25"
# ]

# for col in numeric_columns:
#     if col in merged_df.columns:
#         merged_df[col] = merged_df[col].fillna(merged_df[col].median())


# # Colonnes utiles
# final_columns = [
#     "flow_rate",
#     "occupancy_rate",
#     "temperature",
#     "humidity",
#     "wind_speed",
#     "precipitation",
#     "no2",
#     "pm10",
#     "pm25",
#     "hour",
#     "day_of_week",
#     "month",
#     "is_weekend",
#     "congestion_target"
# ]

# ml_df = merged_df[final_columns]

# print("Valeurs congestion_level :")
# print(merged_df["congestion_level"].value_counts(dropna=False))

# print("Valeurs congestion_target :")
# print(merged_df["congestion_target"].value_counts(dropna=False))

# # Export CSV
# ml_df.to_csv(
#     "data/processed/ml_dataset.csv",
#     index=False
# )

# print("Dataset ML créé avec succès")
# print(ml_df.head())

# print(f"Shape dataset : {ml_df.shape}")

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

print("Chargement des données...")

traffic_df = pd.read_sql("SELECT * FROM traffic_data", engine)
weather_df = pd.read_sql("SELECT * FROM weather_data", engine)
pollution_df = pd.read_sql("SELECT * FROM pollution_data", engine)

print("traffic_df:", traffic_df.shape)
print("weather_df:", weather_df.shape)
print("pollution_df:", pollution_df.shape)

traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"])
weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"])

traffic_df["timestamp_hour"] = traffic_df["timestamp"].dt.floor("h")
weather_df["timestamp_hour"] = weather_df["timestamp"].dt.floor("h")
pollution_df["timestamp_hour"] = pollution_df["timestamp"].dt.floor("h")

print("Traffic timestamp range:", traffic_df["timestamp_hour"].min(), traffic_df["timestamp_hour"].max())
print("Weather timestamp range:", weather_df["timestamp_hour"].min(), weather_df["timestamp_hour"].max())
print("Pollution timestamp range:", pollution_df["timestamp_hour"].min(), pollution_df["timestamp_hour"].max())

merged_df = traffic_df.merge(
    weather_df,
    on="timestamp_hour",
    how="left",
    suffixes=("", "_weather")
)

print("Après fusion météo:", merged_df.shape)

merged_df = merged_df.merge(
    pollution_df,
    on="timestamp_hour",
    how="left",
    suffixes=("", "_pollution")
)

print("Après fusion pollution:", merged_df.shape)

merged_df["flow_rate"] = pd.to_numeric(merged_df["flow_rate"], errors="coerce")
merged_df["occupancy_rate"] = pd.to_numeric(merged_df["occupancy_rate"], errors="coerce")

print("flow_rate non null:", merged_df["flow_rate"].notna().sum())
print("flow_rate exemples:")
print(merged_df["flow_rate"].head(20))

merged_df = merged_df.dropna(subset=["flow_rate"])

# Features temporelles
merged_df["hour"] = merged_df["timestamp_hour"].dt.hour
merged_df["day_of_week"] = merged_df["timestamp_hour"].dt.dayofweek
merged_df["month"] = merged_df["timestamp_hour"].dt.month
merged_df["is_weekend"] = (merged_df["day_of_week"] >= 5).astype(int)

# Remplissage météo/pollution
numeric_columns = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "no2",
    "pm10",
    "pm25",
    "occupancy_rate"
]

for col in numeric_columns:
    if col in merged_df.columns:
        if merged_df[col].notna().sum() == 0:
            merged_df[col] = 0
        else:
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())

def classify_congestion(flow):
    if flow < 300:
        return 0
    elif flow < 800:
        return 1
    else:
        return 2

merged_df["congestion_target"] = merged_df["flow_rate"].apply(classify_congestion)

final_columns = [
    "flow_rate",
    "occupancy_rate",
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "no2",
    "pm10",
    "pm25",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "congestion_target"
]

ml_df = merged_df[final_columns]

print("Valeurs target:")
print(ml_df["congestion_target"].value_counts(dropna=False))

print("Shape final:", ml_df.shape)

ml_df.to_csv("data/processed/ml_dataset.csv", index=False)

print("Dataset ML créé avec succès")
print(ml_df.head())