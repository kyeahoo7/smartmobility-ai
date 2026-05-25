import os
import pandas as pd
import requests

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# =========================
# CONFIGURATION
# =========================

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

print("DB_HOST =", DB_HOST)
print("DB_PORT =", DB_PORT)
print("DB_NAME =", DB_NAME)
print("DB_USER =", DB_USER)
print("DB_PASSWORD =", DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/comptages-routiers-permanents/records?limit=100"
)


# =========================
# EXTRACTION
# =========================

print("Téléchargement des données trafic...")

response = requests.get(URL)

if response.status_code != 200:
    raise Exception(f"Erreur API OpenData Paris : {response.status_code}")

data = response.json()
records = data.get("results", [])

df = pd.json_normalize(records)

print(f"{len(df)} lignes récupérées")
print("Colonnes disponibles :")
print(df.columns)


# =========================
# TRANSFORMATION
# =========================

selected_columns = [
    "q",                  # débit
    "k",                  # taux d'occupation
    "etat_trafic",         # état trafic, souvent vide
    "date_debut",          # timestamp
    "geo_point_2d.lat",
    "geo_point_2d.lon",
    "libelle"
]

df = df[selected_columns]

df.columns = [
    "flow_rate",
    "occupancy_rate",
    "congestion_level",
    "timestamp",
    "latitude",
    "longitude",
    "location_name"
]

# Conversions robustes
df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce")
df["occupancy_rate"] = pd.to_numeric(df["occupancy_rate"], errors="coerce")
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Nettoyage ciblé : on ne supprime pas à cause de etat_trafic vide
df = df.dropna(
    subset=[
        "flow_rate",
        "timestamp",
        "latitude",
        "longitude",
        "location_name"
    ]
)

df["congestion_level"] = df["congestion_level"].fillna("unknown")

# Si occupancy_rate est vide, on le calcule à partir du débit
if df["occupancy_rate"].isna().all():
    max_flow = df["flow_rate"].max()

    if pd.isna(max_flow) or max_flow == 0:
        df["occupancy_rate"] = 0
    else:
        df["occupancy_rate"] = (df["flow_rate"] / max_flow) * 100
else:
    df["occupancy_rate"] = df["occupancy_rate"].fillna(0)

print("Données nettoyées")
print("Nombre de lignes après nettoyage :", len(df))
print(df.head())

# Backup CSV
df.to_csv("data/raw/traffic_data.csv", index=False)


# =========================
# CHARGEMENT LOCATIONS
# =========================

locations_df = df[
    [
        "location_name",
        "latitude",
        "longitude"
    ]
].drop_duplicates()

locations_df["road_name"] = locations_df["location_name"]
locations_df["city"] = "Paris"

locations_df = locations_df[
    [
        "location_name",
        "latitude",
        "longitude",
        "road_name",
        "city"
    ]
]

# Option simple MVP : éviter les doublons en vidant les tables trafic/locations
# Important : on garde l'ordre à cause des clés étrangères
with engine.begin() as conn:
    conn.execute(text("DELETE FROM traffic_data"))

locations_df.to_sql(
    "locations",
    engine,
    if_exists="append",
    index=False
)

print("Locations insérées :", len(locations_df))


# =========================
# CHARGEMENT TRAFFIC DATA
# =========================

locations_db = pd.read_sql(
    """
    SELECT id, location_name, latitude, longitude
    FROM locations
    """,
    engine
)

merged_df = df.merge(
    locations_db,
    on=[
        "location_name",
        "latitude",
        "longitude"
    ],
    how="left"
)

traffic_df = merged_df[
    [
        "id",
        "timestamp",
        "flow_rate",
        "occupancy_rate",
        "congestion_level"
    ]
]

traffic_df.columns = [
    "location_id",
    "timestamp",
    "flow_rate",
    "occupancy_rate",
    "congestion_level"
]

traffic_df = traffic_df.dropna(
    subset=[
        "location_id",
        "timestamp",
        "flow_rate"
    ]
)

traffic_df.to_sql(
    "traffic_data",
    engine,
    if_exists="append",
    index=False
)

print("Traffic data insérées avec succès :", len(traffic_df))

check_df = pd.read_sql(
    "SELECT COUNT(*) AS total FROM traffic_data",
    engine
)

print("Nombre de lignes dans traffic_data :", check_df["total"].iloc[0])