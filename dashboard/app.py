import os
import joblib
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import math

from dotenv import load_dotenv
from sqlalchemy import create_engine


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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


@st.cache_data
def load_data():
    traffic = pd.read_sql("""
        SELECT 
            t.id,
            t.location_id,
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

    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    pollution["timestamp"] = pd.to_datetime(pollution["timestamp"])

    return traffic, weather, pollution


traffic_df, weather_df, pollution_df = load_data()


def congestion_from_flow(flow):
    if flow < 300:
        return "Faible"
    elif flow < 800:
        return "Moyenne"
    return "Forte"


def congestion_score(row):
    return (row["flow_rate"] * 0.7) + (row["occupancy_rate"] * 30)


def get_sensor_code(location_name):
    return pd.Categorical(traffic_df["location_name"]).categories.get_loc(location_name)


def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def get_nearby_transports(lat, lon, radius=1200):
    overpass_url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["railway"="station"];
      node(around:{radius},{lat},{lon})["railway"="tram_stop"];
      node(around:{radius},{lat},{lon})["station"="subway"];
      node(around:{radius},{lat},{lon})["subway"="yes"];
      node(around:{radius},{lat},{lon})["highway"="bus_stop"];
      node(around:{radius},{lat},{lon})["public_transport"];
    );
    out body 50;
    """

    try:
        response = requests.post(
            overpass_url,
            data={"data": query},
            timeout=25
        )

        if response.status_code != 200:
            return []

        elements = response.json().get("elements", [])
        transports = []

        for el in elements:
            tags = el.get("tags", {})

            name = tags.get("name")
            if not name:
                continue

            stop_lat = el.get("lat")
            stop_lon = el.get("lon")

            if stop_lat is None or stop_lon is None:
                continue

            if tags.get("railway") == "station" or tags.get("station") == "subway" or tags.get("subway") == "yes":
                transport_type = "Métro / RER"
            elif tags.get("railway") == "tram_stop":
                transport_type = "Tramway"
            elif tags.get("highway") == "bus_stop":
                transport_type = "Bus"
            else:
                transport_type = "Transport public"

            distance_m = haversine_distance(
                lat,
                lon,
                stop_lat,
                stop_lon
            )

            walking_minutes = round(distance_m / 80)

            transports.append({
                "name": name,
                "type": transport_type,
                "distance_m": round(distance_m),
                "walking_minutes": walking_minutes
            })

        transports = sorted(
            transports,
            key=lambda x: x["distance_m"]
        )

        unique = []
        seen = set()

        for item in transports:
            key = (item["name"], item["type"])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique[:8]

    except Exception:
        return []


def generate_mistral_recommendation(
    location_name,
    congestion,
    current_flow,
    predicted_flow,
    transports
):
    if not MISTRAL_API_KEY:
        return (
            "Clé Mistral absente. Ajoute MISTRAL_API_KEY dans ton fichier .env "
            "pour générer une recommandation IA."
        )

    transports_text = "\n".join(
    [
        f"- {t['name']} ({t['type']}) à {t['distance_m']} m, environ {t['walking_minutes']} min à pied"
        for t in transports
    ]
    )

    prompt = f"""
Tu es un assistant mobilité urbaine pour Paris.

Une congestion est prévue sur le capteur suivant :
- Zone / capteur : {location_name}
- Congestion prévue : {congestion}
- Débit actuel : {round(current_flow, 2)}
- Débit prévu H+1 : {round(predicted_flow, 2)}

Transports en commun proches détectés :
{transports_text}

Génère une recommandation courte, claire et professionnelle.
Objectif :
- expliquer le risque de congestion ;
- citer le transport le plus proche avec sa distance et son temps à pied ;
- proposer une alternative claire pour éviter la zone congestionnée ;
- ne pas inventer de lignes ou stations non listées ;
- donner une recommandation opérationnelle.
"""

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            return f"Erreur Mistral API : {response.status_code} - {response.text}"

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Erreur Mistral : {e}"


traffic_df["congestion_estimee"] = traffic_df["flow_rate"].apply(congestion_from_flow)
traffic_df["criticite_score"] = traffic_df.apply(congestion_score, axis=1)


st.sidebar.title("🚦 SmartMobility Paris AI")
st.sidebar.caption("Dashboard d’analyse et de prévision de mobilité urbaine")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📊 Analyse trafic",
        "🗺️ Carte interactive",
        "🚨 Zones critiques",
        "🌦️ Météo & pollution",
        "🚗 Prévision trafic H+1",
        "📈 Monitoring IA",
        "💡 Recommandations"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "MVP Big Data & IA basé sur des données ouvertes : trafic, météo, pollution et prévision H+1."
)


if page == "🏠 Accueil":
    st.title("🚦 SmartMobility Paris AI")
    st.subheader("Plateforme intelligente d’analyse de la mobilité urbaine à Paris")

    st.markdown("""
    Ce dashboard permet d’analyser les flux de circulation, d’identifier les zones critiques,
    de visualiser les données environnementales et de prédire le trafic à H+1.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lignes trafic", len(traffic_df))
    col2.metric("Capteurs", traffic_df["location_name"].nunique())
    col3.metric("Débit moyen", round(traffic_df["flow_rate"].mean(), 2))
    col4.metric("Occupation moyenne", f"{round(traffic_df['occupancy_rate'].mean(), 2)} %")

    col1, col2, col3 = st.columns(3)
    col1.metric("Débit max", round(traffic_df["flow_rate"].max(), 2))
    col2.metric("Zone la plus chargée", traffic_df.groupby("location_name")["flow_rate"].mean().idxmax())
    col3.metric("Congestion dominante", traffic_df["congestion_estimee"].mode()[0])

    st.divider()

    st.subheader("Pipeline du projet")
    st.code("""
OpenData Paris / Open-Meteo
        ↓
ETL Python
        ↓
PostgreSQL
        ↓
Feature Engineering
        ↓
Modèle IA RandomForest Regressor
        ↓
Dashboard Streamlit
    """)

    st.subheader("Aperçu des données")
    st.dataframe(traffic_df.head(20), width="stretch")


elif page == "📊 Analyse trafic":
    st.title("📊 Analyse du trafic")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            traffic_df,
            x="flow_rate",
            nbins=30,
            title="Distribution du débit de circulation"
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig = px.histogram(
            traffic_df,
            x="occupancy_rate",
            nbins=30,
            title="Distribution du taux d’occupation"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Évolution temporelle du trafic")

    fig = px.line(
        traffic_df.sort_values("timestamp"),
        x="timestamp",
        y="flow_rate",
        color="location_name",
        title="Débit de circulation par capteur"
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Analyse par zone")

    selected_zone = st.selectbox(
        "Choisir une zone",
        sorted(traffic_df["location_name"].dropna().unique())
    )

    zone_df = traffic_df[traffic_df["location_name"] == selected_zone]

    col1, col2, col3 = st.columns(3)
    col1.metric("Débit moyen", round(zone_df["flow_rate"].mean(), 2))
    col2.metric("Occupation moyenne", f"{round(zone_df['occupancy_rate'].mean(), 2)} %")
    col3.metric("État dominant", zone_df["congestion_level"].mode()[0])

    st.dataframe(zone_df, width="stretch")


elif page == "🗺️ Carte interactive":
    st.title("🗺️ Carte interactive des capteurs")

    map_df = (
        traffic_df
        .groupby(["location_name", "latitude", "longitude"])
        .agg(
            flow_rate=("flow_rate", "mean"),
            occupancy_rate=("occupancy_rate", "mean"),
            criticite_score=("criticite_score", "mean")
        )
        .reset_index()
    )

    map_df["congestion_estimee"] = map_df["flow_rate"].apply(congestion_from_flow)

    congestion_filter = st.multiselect(
        "Filtrer par congestion",
        ["Faible", "Moyenne", "Forte"],
        default=["Faible", "Moyenne", "Forte"]
    )

    map_df = map_df[map_df["congestion_estimee"].isin(congestion_filter)]

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        size="flow_rate",
        color="congestion_estimee",
        hover_name="location_name",
        hover_data={
            "flow_rate": ":.2f",
            "occupancy_rate": ":.2f",
            "criticite_score": ":.2f",
            "latitude": False,
            "longitude": False
        },
        zoom=11,
        height=700,
        title="Carte des capteurs de trafic"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, width="stretch")


elif page == "🚨 Zones critiques":
    st.title("🚨 Analyse des zones critiques")

    critical_df = (
        traffic_df
        .groupby("location_name")
        .agg(
            debit_moyen=("flow_rate", "mean"),
            debit_max=("flow_rate", "max"),
            occupation_moyenne=("occupancy_rate", "mean"),
            criticite_score=("criticite_score", "mean"),
            nb_mesures=("id", "count")
        )
        .sort_values("criticite_score", ascending=False)
        .reset_index()
    )

    st.subheader("Top 10 des zones les plus critiques")

    top10 = critical_df.head(10)

    fig = px.bar(
        top10,
        x="location_name",
        y="criticite_score",
        title="Classement des zones critiques",
        text_auto=True
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Tableau détaillé")
    st.dataframe(critical_df, width="stretch")

    st.subheader("Interprétation")
    st.markdown("""
    Le score de criticité combine le débit moyen de circulation et le taux d’occupation moyen.
    Il permet d’identifier rapidement les capteurs où la circulation est la plus sensible.
    """)


elif page == "🌦️ Météo & pollution":
    st.title("🌦️ Météo & pollution")

    st.subheader("Indicateurs météo")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Température moyenne", f"{round(weather_df['temperature'].mean(), 2)} °C")
    col2.metric("Humidité moyenne", f"{round(weather_df['humidity'].mean(), 2)} %")
    col3.metric("Vent moyen", f"{round(weather_df['wind_speed'].mean(), 2)} km/h")
    col4.metric("Précipitations moyennes", round(weather_df["precipitation"].mean(), 2))

    fig = px.line(
        weather_df,
        x="timestamp",
        y=["temperature", "humidity", "wind_speed", "precipitation"],
        title="Évolution des données météo"
    )
    st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("Indicateurs pollution")

    col1, col2, col3 = st.columns(3)
    col1.metric("NO2 moyen", round(pollution_df["no2"].mean(), 2))
    col2.metric("PM10 moyen", round(pollution_df["pm10"].mean(), 2))
    col3.metric("PM2.5 moyen", round(pollution_df["pm25"].mean(), 2))

    fig = px.line(
        pollution_df,
        x="timestamp",
        y=["no2", "pm10", "pm25"],
        title="Évolution des polluants"
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Table pollution")
    st.dataframe(pollution_df.head(50), width="stretch")


elif page == "🚗 Prévision trafic H+1":
    st.title("🚗 Prévision du trafic à H+1")

    st.markdown("""
    Cette section utilise un modèle de régression RandomForest pour prédire le débit de circulation
    à la prochaine mesure du capteur.
    """)

    model_path = "models/traffic_forecast_model.pkl"

    if not os.path.exists(model_path):
        st.error("Modèle forecast introuvable. Lance : python models/train_forecast_model.py")
    else:
        model = joblib.load(model_path)

        selected_zone = st.selectbox(
            "Choisir un capteur",
            sorted(traffic_df["location_name"].dropna().unique())
        )

        zone_df = (
            traffic_df[traffic_df["location_name"] == selected_zone]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if len(zone_df) < 2:
            st.warning("Pas assez de données pour ce capteur.")
        else:
            latest = zone_df.iloc[-1]
            previous = zone_df.iloc[-2]

            st.subheader("Dernière mesure connue")

            col1, col2, col3 = st.columns(3)
            col1.metric("Débit actuel", round(latest["flow_rate"], 2))
            col2.metric("Occupation actuelle", f"{round(latest['occupancy_rate'], 2)} %")
            col3.metric("État actuel", latest["congestion_estimee"])

            input_data = pd.DataFrame([{
                "sensor_code": get_sensor_code(latest["location_name"]),
                "flow_rate": latest["flow_rate"],
                "occupancy_rate": latest["occupancy_rate"],
                "flow_rate_lag_1": previous["flow_rate"],
                "occupancy_lag_1": previous["occupancy_rate"],
                "hour": latest["timestamp"].hour,
                "day_of_week": latest["timestamp"].dayofweek,
                "month": latest["timestamp"].month,
                "is_weekend": int(latest["timestamp"].dayofweek >= 5)
            }])

            predicted_flow = model.predict(input_data)[0]
            predicted_congestion = congestion_from_flow(predicted_flow)
            evolution = predicted_flow - latest["flow_rate"]

            st.subheader("Prévision H+1")

            col1, col2, col3 = st.columns(3)
            col1.metric("Débit prédit", round(predicted_flow, 2))
            col2.metric("Congestion prévue", predicted_congestion)
            col3.metric("Évolution estimée", round(evolution, 2))

            if predicted_congestion == "Forte":
                st.error("⚠️ Risque élevé de congestion sur ce capteur.")
            elif predicted_congestion == "Moyenne":
                st.warning("🟠 Trafic modéré à surveiller.")
            else:
                st.success("🟢 Trafic fluide prévu.")

            if predicted_congestion in ["Moyenne", "Forte"]:
                st.subheader("🤖 Assistant mobilité IA")

                if st.button("Générer une recommandation alternative"):
                    transports = get_nearby_transports(
                        latest["latitude"],
                        latest["longitude"]
                    )

                    st.write("Transports proches détectés :")

                    if transports:
                        st.dataframe(pd.DataFrame(transports), width="stretch")

                    recommendation = generate_mistral_recommendation(
                        location_name=selected_zone,
                        congestion=predicted_congestion,
                        current_flow=latest["flow_rate"],
                        predicted_flow=predicted_flow,
                        transports=transports
                    )

                    st.info(recommendation)

            st.subheader("Historique du capteur")

            fig = px.line(
                zone_df,
                x="timestamp",
                y="flow_rate",
                title=f"Historique du débit - {selected_zone}"
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("Données utilisées par le modèle")
            st.dataframe(input_data, width="stretch")


elif page == "📈 Monitoring IA":
    st.title("📈 Monitoring du modèle IA")

    model_path = "models/traffic_forecast_model.pkl"
    dataset_path = "data/processed/forecast_dataset.csv"

    if not os.path.exists(model_path):
        st.error("Modèle introuvable.")

    elif not os.path.exists(dataset_path):
        st.error("Dataset forecast introuvable.")

    else:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = joblib.load(model_path)
        ml_df = pd.read_csv(dataset_path)

        X = ml_df.drop(columns=["flow_rate_next"])
        y = ml_df["flow_rate_next"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        st.subheader("Performances du modèle")

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(mae, 2))
        col2.metric("RMSE", round(rmse, 2))
        col3.metric("R²", round(r2, 4))

        st.divider()

        st.subheader("Informations dataset")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre de lignes", ml_df.shape[0])
        col2.metric("Nombre de variables", ml_df.shape[1] - 1)
        col3.metric("Variable cible", "flow_rate_next")

        st.divider()

        st.subheader("Comparaison Réalité vs Prédiction")

        comparison_df = pd.DataFrame({
            "Valeur réelle": y_test.values,
            "Prédiction": y_pred
        })

        fig = px.scatter(
            comparison_df,
            x="Valeur réelle",
            y="Prédiction",
            title="Prédictions vs Réalité"
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()

        st.subheader("Distribution de la cible")

        fig = px.histogram(
            ml_df,
            x="flow_rate_next",
            nbins=30,
            title="Distribution du trafic futur"
        )

        st.plotly_chart(fig, width="stretch")

        st.divider()

        st.subheader("Importance des variables")

        feature_names = X.columns

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Variables les plus importantes"
        )

        st.plotly_chart(fig, width="stretch")
        st.dataframe(importance_df, width="stretch")

        st.divider()

        st.subheader("Interprétation métier")

        st.success(f"""
Le modèle explique environ {round(r2*100,2)} % de la variation du trafic futur.

Une erreur moyenne d'environ {round(mae,2)} véhicules est observée.

Les variables les plus influentes sont visibles ci-dessus et permettent
d'anticiper les niveaux futurs de circulation sur les capteurs.
        """)


elif page == "💡 Recommandations":
    st.title("💡 Recommandations intelligentes")

    st.markdown("""
    Cette page identifie les capteurs à risque à partir du modèle de prévision H+1
    et génère des recommandations opérationnelles.
    """)

    model_path = "models/traffic_forecast_model.pkl"

    if not os.path.exists(model_path):
        st.error("Modèle forecast introuvable. Lance : python models/train_forecast_model.py")

    else:
        model = joblib.load(model_path)
        recommendations = []

        for location_name in traffic_df["location_name"].dropna().unique():
            zone_df = (
                traffic_df[traffic_df["location_name"] == location_name]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            if len(zone_df) < 2:
                continue

            latest = zone_df.iloc[-1]
            previous = zone_df.iloc[-2]

            input_data = pd.DataFrame([{
                "sensor_code": get_sensor_code(latest["location_name"]),
                "flow_rate": latest["flow_rate"],
                "occupancy_rate": latest["occupancy_rate"],
                "flow_rate_lag_1": previous["flow_rate"],
                "occupancy_lag_1": previous["occupancy_rate"],
                "hour": latest["timestamp"].hour,
                "day_of_week": latest["timestamp"].dayofweek,
                "month": latest["timestamp"].month,
                "is_weekend": int(latest["timestamp"].dayofweek >= 5)
            }])

            predicted_flow = model.predict(input_data)[0]
            congestion = congestion_from_flow(predicted_flow)
            evolution = predicted_flow - latest["flow_rate"]

            if congestion == "Forte":
                recommendation = "Éviter la zone ou proposer un itinéraire alternatif."
                priority = "Haute"
            elif congestion == "Moyenne":
                recommendation = "Surveiller la zone et anticiper une possible hausse du trafic."
                priority = "Moyenne"
            else:
                recommendation = "Trafic fluide, aucune action urgente nécessaire."
                priority = "Faible"

            recommendations.append({
                "Capteur": location_name,
                "Débit actuel": round(latest["flow_rate"], 2),
                "Débit prévu H+1": round(predicted_flow, 2),
                "Évolution": round(evolution, 2),
                "Congestion prévue": congestion,
                "Priorité": priority,
                "Recommandation": recommendation
            })

        reco_df = pd.DataFrame(recommendations)

        if reco_df.empty:
            st.warning("Pas assez de données pour générer des recommandations.")

        else:
            st.subheader("Synthèse des recommandations")

            col1, col2, col3 = st.columns(3)
            col1.metric("Alertes haute priorité", len(reco_df[reco_df["Priorité"] == "Haute"]))
            col2.metric("Alertes moyenne priorité", len(reco_df[reco_df["Priorité"] == "Moyenne"]))
            col3.metric("Zones fluides", len(reco_df[reco_df["Priorité"] == "Faible"]))

            st.divider()

            priority_filter = st.multiselect(
                "Filtrer par priorité",
                ["Haute", "Moyenne", "Faible"],
                default=["Haute", "Moyenne", "Faible"]
            )

            filtered_df = reco_df[reco_df["Priorité"].isin(priority_filter)]

            st.dataframe(
                filtered_df.sort_values("Évolution", ascending=False),
                width="stretch"
            )

            st.divider()

            st.subheader("Top zones à surveiller")

            top_risk = filtered_df.sort_values(
                "Débit prévu H+1",
                ascending=False
            ).head(10)

            fig = px.bar(
                top_risk,
                x="Capteur",
                y="Débit prévu H+1",
                color="Priorité",
                title="Top 10 des zones à risque selon la prévision H+1",
                text_auto=True
            )

            st.plotly_chart(fig, width="stretch")

            st.divider()

            st.subheader("Lecture métier")

            st.info("""
Cette page transforme les prédictions du modèle en décisions opérationnelles.
Elle permet d’identifier rapidement les zones à surveiller et de proposer
des actions adaptées selon le niveau de congestion attendu.
            """)