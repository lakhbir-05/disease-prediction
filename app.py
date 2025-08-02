
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# Load the model and utilities
# ============================
model = joblib.load("disease_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
symptoms = joblib.load("symptoms_list.pkl")

# Medicine mapping
medicine_dict = {
    "Fungal infection": "Fluconazole",
    "Allergy": "Cetirizine",
    "GERD": "Omeprazole",
    "Chronic cholestasis": "Ursodeoxycholic acid",
    "Drug Reaction": "Antihistamines",
    "Peptic ulcer disease": "Pantoprazole",
    "AIDS": "Antiretroviral Therapy",
    "Diabetes": "Metformin",
    "Hypertension": "Amlodipine",
    "Migraine": "Sumatriptan",
    # Add more mappings
}

# ============================
# Streamlit App UI
# ============================
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("🩺 Disease Prediction & Medicine Recommendation")

# ----------------------------
# Feature 4: User Info Section
# ----------------------------
with st.expander("👤 Optional: Enter Your Information"):
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])
    location = st.text_input("Your Location (optional)")

# ----------------------------
# Feature 2: Symptom Search/Filter (already in multiselect, we sort alphabetically)
# ----------------------------
st.subheader("🩹 Select Symptoms")
selected_symptoms = st.multiselect("Choose your symptoms:", sorted(symptoms))

# ----------------------------
# Feature 5: Symptom Severity
# ----------------------------
severity_dict = {}
if selected_symptoms:
    st.subheader("📊 Symptom Severity (1 = mild, 5 = severe)")
    for symptom in selected_symptoms:
        severity_dict[symptom] = st.slider(symptom, 1, 5, 3)

# ----------------------------
# Feature 7: CSV Upload
# ----------------------------
uploaded_file = st.file_uploader("📁 Or Upload Symptom Data CSV", type=["csv"])

if uploaded_file:
    uploaded_df = pd.read_csv(uploaded_file)
    if all(symptom in uploaded_df.columns for symptom in symptoms):
        input_data = uploaded_df[symptoms].iloc[0].values.reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        predicted_encoded = model.predict(input_scaled)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_encoded])[0]
        st.success(f"📋 Uploaded CSV Prediction: **{predicted_disease}**")
    else:
        st.warning("CSV must contain columns matching the full symptom list.")

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔮 Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_data = []
        for symptom in symptoms:
            if symptom in severity_dict:
                # Use severity (1-5) for selected, else 0
                input_data.append(severity_dict[symptom])
            else:
                input_data.append(0)
        input_data = np.array(input_data).reshape(1, -1)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        predicted_encoded = model.predict(input_scaled)[0]
        predicted_disease = label_encoder.inverse_transform([predicted_encoded])[0]

        st.success(f"**Predicted Disease:** {predicted_disease}")

        # Recommended medicine
        recommended_medicine = medicine_dict.get(predicted_disease, "Consult a doctor for appropriate medicine.")
        st.info(f"**Recommended Medicine:** {recommended_medicine}")

        # Show probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            prob_df = pd.DataFrame({
                "Disease": label_encoder.inverse_transform(np.arange(len(probs))),
                "Probability": probs
            }).sort_values(by="Probability", ascending=False).head(5)
            st.subheader("🧪 Top 5 Predictions")
            st.dataframe(prob_df)

