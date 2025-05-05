# crop_yield_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -------------------------------
# Generate synthetic dataset
# -------------------------------
@st.cache_data
def create_model():
    np.random.seed(42)
    n_samples = 500
    crops = ['Wheat', 'Rice', 'Maize']
    soils = ['Loamy', 'Clay', 'Sandy']
    regions = ['North', 'South', 'East', 'West']

    data = pd.DataFrame({
        'Crop': np.random.choice(crops, n_samples),
        'Soil Type': np.random.choice(soils, n_samples),
        'Region': np.random.choice(regions, n_samples),
        'Rainfall': np.random.uniform(100, 300, n_samples),
        'Temperature': np.random.uniform(20, 35, n_samples),
        'Fertilizer': np.random.uniform(20, 100, n_samples),
    })

    data['Yield (kg/ha)'] = (
        20 * (data['Rainfall'] / 100)
        + 10 * (data['Fertilizer'] / 50)
        - 5 * (data['Temperature'] / 25)
        + np.random.normal(0, 3, n_samples)
    )

    # Encode categorical columns
    label_encoders = {}
    for col in ['Crop', 'Soil Type', 'Region']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data.drop(columns=['Yield (kg/ha)'])
    y = data['Yield (kg/ha)']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, label_encoders

# Load model and encoders
model, label_encoders = create_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌾 Crop Yield Prediction App")

# Inputs
crop = st.selectbox("Select Crop", ['Wheat', 'Rice', 'Maize'])
soil = st.selectbox("Select Soil Type", ['Loamy', 'Clay', 'Sandy'])
region = st.selectbox("Select Region", ['North', 'South', 'East', 'West'])

rainfall = st.slider("Rainfall (mm)", 100, 300, 180)
temperature = st.slider("Temperature (°C)", 15, 40, 28)
fertilizer = st.slider("Fertilizer Used (kg/ha)", 10, 100, 50)

# Prediction
if st.button("Predict Yield"):
    input_df = pd.DataFrame([{
        'Crop': label_encoders['Crop'].transform([crop])[0],
        'Soil Type': label_encoders['Soil Type'].transform([soil])[0],
        'Region': label_encoders['Region'].transform([region])[0],
        'Rainfall': rainfall,
        'Temperature': temperature,
        'Fertilizer': fertilizer,
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Estimated Crop Yield: **{prediction:.2f} kg/ha**")
