import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def feature_engineering(df):

    df = df.copy()

    # Convert TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # =========================
    # NEW FEATURES
    # =========================
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['engagement_score'] = df['tenure'] * df['MonthlyCharges']
    df['high_value_customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    # =========================
    # TARGET (SAFE FIX ✅)
    # =========================
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

    # =========================
    # ENCODING
    # =========================
    df = pd.get_dummies(df, drop_first=True)

    return df