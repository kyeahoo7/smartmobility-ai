import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
from sqlalchemy import create_engine


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="SmartMobility Paris AI",
    page_icon="🚦",
    layout="wide"
)

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    traffic = pd.read_sql("""
        SELECT 
            t.id,
            t.timestamp,
            t.flow_rate,
            t.occupancy_rate,
            t.congestion_level,
            l.location_name,
            l.latitude,
            l.longitude
        FROM traffic_data t
        LEFT JOIN locations l ON t.location_id = l.id
    """, engine)

    weather = pd.read_sql("SELECT * FROM weather_data", engine)
    pollution = pd.read_sql("SELECT * FROM pollution_data", engine)

    return traffic, weather, pollution


traffic_df, weather_df, pollution_df = load_data()


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🚦 SmartMobility Paris AI")
page = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Analyse trafic",
        "Carte interactive",
        "Météo & pollution",
        "Prédiction IA"
    ]
)


# =========================
# HELPERS
# =========================

def classify_congestion_label(value):
    if value == 0:
        return "Faible"
    elif value == 1:
        return "Moyenne"
    elif value == 2:
        return "Forte"
    else:
        return "Inconnue"


def classify_color_label(flow):
    if flow < 300:
        return "Faible"
    elif flow < 800:
        return "Moyenne"
    else:
        return "Forte"


# =========================
# ACCUEIL
# =========================

if page == "Accueil":
    st.title("🚦 SmartMobility Paris AI")
    st.subheader("Plateforme intelligente d’analyse et de prédiction de la mobilité urbaine")

    st.markdown("""
    Ce dashboard présente un MVP de plateforme data/IA pour l’analyse du trafic parisien.
    
    Il combine :
    - données de trafic routier,
    - données météo,
    - données pollution,
    - modèle IA de prédiction de congestion.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Lignes trafic", len(traffic_df))
    col2.metric("Zones / capteurs", traffic_df["location_name"].nunique())
    col3.metric("Débit moyen", round(traffic_df["flow_rate"].mean(), 2))
    col4.metric("Occupation moyenne", f"{round(traffic_df['occupancy_rate'].mean(), 2)} %")

    st.divider()

    st.subheader("Aperçu des données trafic")
    st.dataframe(traffic_df.head(20), use_container_width=True)


# =========================
# ANALYSE TRAFIC
# =========================

elif page == "Analyse trafic":
    st.title("📊 Analyse du trafic")

    st.subheader("Distribution du débit de circulation")

    fig = px.histogram(
        traffic_df,
        x="flow_rate",
        nbins=30,
        title="Distribution du flow_rate"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 des zones avec le plus fort débit moyen")

    top_zones = (
        traffic_df
        .groupby("location_name")["flow_rate"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_zones,
        x="location_name",
        y="flow_rate",
        title="Top zones par débit moyen"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Débit et occupation par zone")

    selected_zone = st.selectbox(
        "Choisir une zone",
        traffic_df["location_name"].dropna().unique()
    )

    zone_df = traffic_df[traffic_df["location_name"] == selected_zone]

    col1, col2 = st.columns(2)
    col1.metric("Débit moyen zone", round(zone_df["flow_rate"].mean(), 2))
    col2.metric("Occupation moyenne zone", f"{round(zone_df['occupancy_rate'].mean(), 2)} %")

    st.dataframe(zone_df, use_container_width=True)


# =========================
# CARTE INTERACTIVE
# =========================

elif page == "Carte interactive":
    st.title("🗺️ Carte interactive des zones de trafic")

    map_df = (
        traffic_df
        .groupby(["location_name", "latitude", "longitude"])
        .agg(
            flow_rate=("flow_rate", "mean"),
            occupancy_rate=("occupancy_rate", "mean")
        )
        .reset_index()
    )

    map_df["congestion"] = map_df["flow_rate"].apply(classify_color_label)

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        size="flow_rate",
        color="congestion",
        hover_name="location_name",
        hover_data={
            "flow_rate": True,
            "occupancy_rate": True,
            "latitude": False,
            "longitude": False
        },
        zoom=11,
        height=650,
        title="Carte des capteurs de trafic"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# METEO POLLUTION
# =========================

elif page == "Météo & pollution":
    st.title("🌦️ Météo & pollution")

    col1, col2, col3 = st.columns(3)

    if len(weather_df) > 0:
        col1.metric("Température moyenne", f"{round(weather_df['temperature'].mean(), 2)} °C")
        col2.metric("Humidité moyenne", f"{round(weather_df['humidity'].mean(), 2)} %")
        col3.metric("Vent moyen", f"{round(weather_df['wind_speed'].mean(), 2)} km/h")

        st.subheader("Évolution météo")
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        fig = px.line(
            weather_df,
            x="timestamp",
            y=["temperature", "humidity", "wind_speed"],
            title="Météo horaire"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if len(pollution_df) > 0:
        col1, col2, col3 = st.columns(3)

        col1.metric("NO2 moyen", round(pollution_df["no2"].mean(), 2))
        col2.metric("PM10 moyen", round(pollution_df["pm10"].mean(), 2))
        col3.metric("PM2.5 moyen", round(pollution_df["pm25"].mean(), 2))

        st.subheader("Évolution pollution")
        pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"])

        fig = px.line(
            pollution_df,
            x="timestamp",
            y=["no2", "pm10", "pm25"],
            title="Pollution horaire"
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================
# PREDICTION IA
# =========================

elif page == "Prédiction IA":
    st.title("🤖 Prédiction de congestion")

    st.markdown("""
    Cette section utilise le modèle RandomForest entraîné pour prédire un niveau de congestion.
    
    Classes :
    - 0 : faible
    - 1 : moyenne
    - 2 : forte
    """)

    model_path = "models/random_forest_model.pkl"

    if not os.path.exists(model_path):
        st.error("Modèle introuvable. Lance d'abord : python models/train_model.py")
    else:
        model = joblib.load(model_path)

        col1, col2 = st.columns(2)

        with col1:
            flow_rate = st.slider("Débit trafic", 0, 6000, 1200)
            occupancy_rate = st.slider("Taux d'occupation (%)", 0.0, 100.0, 20.0)
            temperature = st.slider("Température", -10.0, 40.0, 18.0)
            humidity = st.slider("Humidité", 0.0, 100.0, 50.0)
            wind_speed = st.slider("Vitesse du vent", 0.0, 100.0, 10.0)

        with col2:
            precipitation = st.slider("Précipitations", 0.0, 50.0, 0.0)
            no2 = st.slider("NO2", 0.0, 150.0, 20.0)
            pm10 = st.slider("PM10", 0.0, 150.0, 15.0)
            pm25 = st.slider("PM2.5", 0.0, 150.0, 10.0)
            hour = st.slider("Heure", 0, 23, 8)
            day_of_week = st.slider("Jour de semaine", 0, 6, 1)
            month = st.slider("Mois", 1, 12, 5)
            is_weekend = st.selectbox("Week-end ?", [0, 1])

        input_data = pd.DataFrame([{
            "flow_rate": flow_rate,
            "occupancy_rate": occupancy_rate,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
            "no2": no2,
            "pm10": pm10,
            "pm25": pm25,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend
        }])

        if st.button("Prédire la congestion"):
            prediction = model.predict(input_data)[0]
            label = classify_congestion_label(prediction)

            st.success(f"Niveau de congestion prédit : {label}")

            st.dataframe(input_data, use_container_width=True)