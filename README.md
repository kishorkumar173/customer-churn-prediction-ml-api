# 🚀 Customer Churn Prediction System

## 📌 Overview
This project is an end-to-end Machine Learning system that predicts whether a customer is likely to churn based on their behavior and subscription details.

It includes:
- Data preprocessing & feature engineering
- Machine Learning model (Random Forest)
- FastAPI backend for real-time predictions
- Streamlit frontend for user interaction
- Deployment on Render

---

## 🎯 Objective
To help businesses identify customers at risk of leaving and take proactive retention actions.

---

## 🧠 How It Works
1. User enters customer details via frontend
2. Data is sent to FastAPI backend
3. Model processes input and predicts churn probability
4. Result is returned and displayed

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Render

---

## 📊 Features
- Predict churn probability in real-time
- User-friendly interface
- API-based architecture
- Industry-style project structure

---

## 🌐 Live Demo

### 🔗 API
https://customer-churn-prediction-ml-api.onrender.com/docs

### 🔗 Frontend
(Add your Streamlit link here)

---

## 📁 Project Structure
Customer-Churn-Prediction/
│
├── data/
├── src/
├── app/ # FastAPI backend
├── frontend/ # Streamlit UI
├── models/
├── outputs/
├── requirements.txt
└── README.md


---

## ▶️ How to Run Locally

### 1. Clone the repo
git clone https://github.com/your-username/customer-churn-prediction-ml-api.git

cd customer-churn-prediction-ml-api

### 2. Install dependencies

pip install -r requirements.txt


### 3. Run backend

uvicorn app.main:app --reload


### 4. Run frontend

streamlit run frontend/app.py


---

## 📈 Sample Output
- Churn Probability: 0.72 → Likely to churn
- Churn Probability: 0.08 → Likely to stay

---

## 💡 Business Use Case
Companies can use this system to:
- Identify high-risk customers
- Offer personalized discounts
- Improve customer retention strategies

---

## 🚀 Future Improvements
- Add SHAP explainability
- Use advanced models (XGBoost)
- Add dashboard analytics
- Integrate real-time data pipeline

---

## 👨‍💻 Author
Kishor Kumar

---

## ⭐ If you like this project, give it a star!