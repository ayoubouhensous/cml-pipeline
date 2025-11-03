import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import joblib

# 1️⃣ Charger les données (jeu de chiffres manuscrits)
digits = load_digits()
X, y = digits.data, digits.target

# 2️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4️⃣ Évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# 5️⃣ Sauvegarder le modèle
joblib.dump(model, 'models/digits_model.pkl')

# 6️⃣ Sauvegarder les métriques
metrics = {
    "accuracy": accuracy,
    "n_estimators": 100,
    "test_size": len(X_test),
    "precision_weighted": report["weighted avg"]["precision"],
    "recall_weighted": report["weighted avg"]["recall"]
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Modèle entraîné avec succès !")
print(f"📊 Accuracy: {accuracy:.4f}")
