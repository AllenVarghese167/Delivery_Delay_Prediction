import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Delivery Delay Predictor")

def load_model():
    model = joblib.load('delay.pkl')
    scaler = joblib.load('scaler.pkl')
    le_traffic = joblib.load('le_traffic.pkl')
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, le_traffic, feature_columns

model, scaler, le_traffic, feature_columns = load_model()

st.header("Enter Delivery Details")

distance = st.slider("Distance in KM", min_value=0, max_value=50, value=5)
experience = st.slider("Experience", min_value=0, max_value=20, value=2)
preparation = st.slider("Preparation Time (min)", min_value=1, max_value=120, value=30)

traffic = st.selectbox("Traffic Level",options=["Low",'Medium','High'])
weather = st.selectbox("Weather",options=['Windy', 'Clear', 'Foggy', 'Rainy', 'Snowy'])
time = st.selectbox("Time Of Day",options=['Afternoon', 'Evening', 'Morning', 'Night'])
vehicle = st.selectbox("Vehicle Type",options=['Bike', 'Car', 'Scooter'])

expected_time = st.number_input(label="Enter Expected Delivery Time",min_value=0,max_value=100,value=0)

if st.button("Predict Delivery Delay"):
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Traffic_Level": le_traffic.transform([traffic]),
        "Weather": [weather],
        "Time_of_Day": [time],
        "Vehicle_Type": [vehicle],
        "Preparation_Time_min": [preparation],
        "Courier_Experience_yrs": [experience]
    })

    categories = {
        "Weather": ["Clear", "Foggy", "Rainy", "Snowy", "Windy"],
        "Time_of_Day": ["Afternoon", "Evening", "Morning", "Night"],
        "Vehicle_Type": ["Bike", "Car", "Scooter"]
    }

    for col, cats in categories.items():
        input_data[col] = pd.Categorical(
            input_data[col], categories=cats
        )

    input_encoded = pd.get_dummies(
        input_data,
        columns=list(categories.keys()),
        drop_first=True,
        dtype=int
    )

    input_encoded = input_encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    features_scaled = scaler.transform(input_encoded)
    prediction = float(model.predict(features_scaled)[0])

    st.metric("Predicted Delivery Time", f"{prediction:.1f} minutes")

    if expected_time > 0:
    
        if prediction > expected_time:
            st.error("Delivery is predicted to be DELAYED.")
        else:
            st.success("Delivery is predicted to be ON TIME.")