import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from feature_engineering import feature_engineering

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/raw/churn.csv")

# =========================
# FEATURE ENGINEERING
# =========================
df = feature_engineering(df)

# =========================
# SPLIT DATA
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 🔥 IMPORTANT for imbalance
)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"   # 🔥 IMPORTANT for churn problem
)

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\n✅ Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL + COLUMNS
# =========================
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("✅ Model + Columns Saved Successfully")