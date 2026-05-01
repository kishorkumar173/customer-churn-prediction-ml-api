import pandas as pd
import joblib
from feature_engineering import feature_engineering

# Load model
model = joblib.load("models/churn_model.pkl")

# Load data
df = pd.read_csv("data/raw/churn.csv")

# Apply same processing
df = feature_engineering(df)

X = df.drop("Churn", axis=1)

# Predict probabilities
df['churn_probability'] = model.predict_proba(X)[:,1]

# Save results
df.to_csv("outputs/predictions.csv", index=False)

print("✅ Predictions Saved")