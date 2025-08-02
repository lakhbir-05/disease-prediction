# disease-prediction
medicine recommendation
# 🩺 Disease Prediction & Medicine Recommendation App

This Streamlit web application predicts possible diseases based on symptoms selected by the user and recommends suitable medicines. It uses a trained machine learning model on a medical symptoms dataset and provides additional features like severity input, CSV upload, and prediction confidence.

---

## 🚀 Features

- ✅ Predict disease based on selected symptoms
- 💊 Recommend medicines for predicted diseases
- 📊 Show model confidence (Top 5 disease probabilities)
- 📂 CSV upload for batch predictions
- 🎚️ Symptom severity input (1–5 scale)
- 👤 Optional user info (age, gender, location)

---

## 📁 Files Included

- `app.py` – Main Streamlit app
- `disease_model.pkl` – Trained Random Forest model
- `scaler.pkl` – StandardScaler used for feature scaling
- `label_encoder.pkl` – Encoder for disease labels
- `symptoms_list.pkl` – List of all symptoms/features
- `requirements.txt` – Python dependencies for the app

---

## 🧠 Model Training

The model was trained using a Random Forest Classifier on a dataset containing 132 symptoms and their associated diseases, originally from [this Kaggle notebook](https://www.kaggle.com/code/miadul/disease-prediction-using-machine-learning/input).

accuracy = 100%
---

## 📦 Installation

### 🔧 Local Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/disease-predictor-app.git
   cd disease-predictor-app
2.Install dependencies

pip install -r requirements.txt

3.Run the app:

streamlit run app.py
