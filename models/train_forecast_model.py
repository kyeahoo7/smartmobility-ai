import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Chargement dataset forecast...")

df = pd.read_csv("data/processed/forecast_dataset.csv")

print(df.head())
print("Shape :", df.shape)

X = df.drop(columns=["flow_rate_next"])
y = df["flow_rate_next"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

print("Entraînement du modèle de prévision H+1...")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Résultats modèle H+1")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

joblib.dump(model, "models/traffic_forecast_model.pkl")

print("Modèle forecast sauvegardé : models/traffic_forecast_model.pkl")