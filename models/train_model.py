import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Chargement dataset ML...")

df = pd.read_csv(
    "data/processed/ml_dataset.csv"
)

print(df.head())

# Features
X = df.drop(columns=["congestion_target"])

# Target
y = df["congestion_target"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Train size : {X_train.shape}")
print(f"Test size : {X_test.shape}")

# Modèle
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

print("Entraînement du modèle...")

model.fit(X_train, y_train)

print("Prédictions...")

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

# Rapport complet
print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Sauvegarde modèle
joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("Modèle sauvegardé")