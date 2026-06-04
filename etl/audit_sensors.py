import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/"
    "datasets/comptages-routiers-permanents/records"
)

LIMIT = 100
MAX_RECORDS = 5000

all_records = []

print("Récupération des données pour audit capteurs...")

for offset in range(0, MAX_RECORDS, LIMIT):
    url = (
        f"{BASE_URL}"
        f"?limit={LIMIT}"
        f"&offset={offset}"
        f"&order_by=t_1h%20desc"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("Erreur API :", response.status_code)
        break

    records = response.json().get("results", [])

    if not records:
        break

    all_records.extend(records)
    print(f"Offset {offset} : {len(records)} lignes")

df = pd.json_normalize(all_records)

print("Total brut :", len(df))

df = df[[
    "libelle",
    "t_1h",
    "q",
    "k",
    "etat_trafic",
    "geo_point_2d.lat",
    "geo_point_2d.lon"
]]

df["timestamp"] = pd.to_datetime(df["t_1h"], errors="coerce")
df["flow_rate"] = pd.to_numeric(df["q"], errors="coerce")
df["occupancy_rate"] = pd.to_numeric(df["k"], errors="coerce")

df = df.dropna(subset=[
    "libelle",
    "timestamp",
    "flow_rate",
    "geo_point_2d.lat",
    "geo_point_2d.lon"
])

audit = (
    df.groupby("libelle")
    .agg(
        nb_mesures=("timestamp", "count"),
        date_min=("timestamp", "min"),
        date_max=("timestamp", "max"),
        debit_moyen=("flow_rate", "mean"),
        occupation_moyenne=("occupancy_rate", "mean")
    )
    .sort_values("nb_mesures", ascending=False)
    .reset_index()
)

audit.to_csv("data/processed/sensor_audit.csv", index=False)

print("\nTop capteurs par nombre de mesures :")
print(audit.head(30))

print("\nFichier créé : data/processed/sensor_audit.csv")