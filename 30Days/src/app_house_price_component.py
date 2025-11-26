# Demo: House price prediction component for AI Insight Hub app using Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

def house_price_app_component():
    st.header("House Price Prediction")
    # Load data for demo
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    # Feature engineering
    df['RoomsPerPop'] = df['AveRooms'] / (df['Population'] + 1e-5)
    df['BedroomRatio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
    X = df.drop('Price', axis=1)
    y = df['Price']
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    # Save model for app
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'house_price_scaler.pkl')
    # User input
    st.subheader("Enter House Features")
    med_inc = st.slider("Median Income", float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()))
    house_age = st.slider("House Age", float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()))
    ave_rooms = st.slider("Average Rooms", float(df['AveRooms'].min()), float(df['AveRooms'].max()), float(df['AveRooms'].mean()))
    ave_bedrms = st.slider("Average Bedrooms", float(df['AveBedrms'].min()), float(df['AveBedrms'].max()), float(df['AveBedrms'].mean()))
    population = st.slider("Population", float(df['Population'].min()), float(df['Population'].max()), float(df['Population'].mean()))
    ave_occup = st.slider("Average Occupancy", float(df['AveOccup'].min()), float(df['AveOccup'].max()), float(df['AveOccup'].mean()))
    latitude = st.slider("Latitude", float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].mean()))
    longitude = st.slider("Longitude", float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].mean()))
    # Compute engineered features
    rooms_per_pop = ave_rooms / (population + 1e-5)
    bedroom_ratio = ave_bedrms / (ave_rooms + 1e-5)
    # Predict
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, rooms_per_pop, bedroom_ratio]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.subheader(f"Predicted House Price: ${prediction[0]:.2f}")

if __name__ == "__main__":
    house_price_app_component()