import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/raw/churn.csv")

print("✅ Data Loaded")
print("Shape:", df.shape)

# Basic info
print("\n--- INFO ---")
print(df.info())

# Missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Target distribution
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts())

# Convert TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop missing
df = df.dropna()

# ==========================
# 📊 VISUALIZATIONS
# ==========================

# Churn Count
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("outputs/churn_distribution.png")
plt.show()

# Tenure vs Churn
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title("Tenure vs Churn")
plt.savefig("outputs/tenure_vs_churn.png")
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("outputs/monthly_vs_churn.png")
plt.show()

# Contract Type vs Churn
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract vs Churn")
plt.xticks(rotation=30)
plt.savefig("outputs/contract_vs_churn.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")
plt.show()

print("✅ EDA Completed")