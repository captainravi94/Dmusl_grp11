import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("🧠 Stroke Prediction System")
st.write("Enter patient details to predict stroke risk")

# User Inputs
age = st.number_input("Age", 1, 120)
avg_glucose = st.number_input("Average Glucose Level", 40.0, 400.0)
bmi = st.number_input("BMI", 10.0, 80.0)
bp_systolic = st.number_input("Systolic BP", 70, 250)
bp_diastolic = st.number_input("Diastolic BP", 40, 150)
stress = st.slider("Stress Level", 0, 10)
sleep = st.slider("Sleep Hours", 0, 12)

if st.button("Predict Stroke Risk"):
    input_data = np.array([[age, avg_glucose, bmi, bp_systolic, bp_diastolic, stress, sleep]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Stroke Risk ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Stroke Risk ({prob*100:.2f}%)")
