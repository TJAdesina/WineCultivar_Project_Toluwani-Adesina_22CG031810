import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Wine Cultivar Predictor")
st.title("🍷 Wine Cultivar Origin Prediction System")

st.write("Enter the chemical properties of the wine sample:")

# -----------------------------
# User Inputs
# -----------------------------
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)
malic_acid = st.number_input("Malic Acid", min_value=0.0, step=0.1)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", min_value=0.0, step=0.1)
magnesium = st.number_input("Magnesium", min_value=0.0, step=1.0)
color_intensity = st.number_input("Color Intensity", min_value=0.0, step=0.1)
proline = st.number_input("Proline", min_value=0.0, step=1.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Cultivar"):
    input_data = np.array([[
        alcohol,
        malic_acid,
        alcalinity_of_ash,
        magnesium,
        color_intensity,
        proline
    ]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.success(f"Predicted Wine Cultivar: Cultivar {prediction + 1}")
