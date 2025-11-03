import os
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_iris  # ⚠️ utiliser Iris comme evaluate.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# === 0️⃣ Définir les répertoires ===
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("📂 cwd        :", os.getcwd())
print("📂 script dir :", BASE_DIR)
print("📂 models     :", MODELS_DIR)
print("📂 reports    :", REPORTS_DIR)

# === 1️⃣ Charger les données ===
iris = load_iris()
X, y = iris.data, iris.target

# === 2️⃣ Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3️⃣ Entraînement du modèle ===
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# === 4️⃣ Évaluation ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === 5️⃣ Sauvegarder le modèle ===
model_path = MODELS_DIR / "digits_model.pkl"
joblib.dump(model, model_path)
print(f"💾 Modèle sauvegardé dans : {model_path}")

# === 6️⃣ Sauvegarder les métriques ===
metrics = {
    "accuracy": accuracy,
    "n_estimators": 50,
    "test_size": len(X_test),
    "precision_weighted": report["weighted avg"]["precision"],
    "recall_weighted": report["weighted avg"]["recall"]
}

metrics_path = REPORTS_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"📊 Fichier de métriques créé : {metrics_path}")

# === 7️⃣ Test de fin ===
test_file = REPORTS_DIR / "DEBUG_TRAIN_OK.txt"
test_file.write_text("train.py exécuté avec succès.\n")

print(f"✅ Modèle entraîné avec succès ! Accuracy = {accuracy:.4f}")
