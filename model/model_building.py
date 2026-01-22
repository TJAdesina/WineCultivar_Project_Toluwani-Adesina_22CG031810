import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("Loading Wine dataset...")
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target

# -----------------------------
# 2. Feature Selection (6 only)
# -----------------------------
selected_features = [
    'alcohol',
    'malic_acid',
    'alcalinity_of_ash',
    'magnesium',
    'color_intensity',
    'proline'
]

X = df[selected_features]
y = df['cultivar']

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 6. Model Evaluation
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save Model & Scaler
# -----------------------------
joblib.dump(model, "wine_cultivar_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")
