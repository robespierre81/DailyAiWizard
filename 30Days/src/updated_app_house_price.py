# Demo: Updated house price prediction app component from Day 61 with evaluation metrics in Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def updated_house_price_app():
    st.header("House Price Prediction with Evaluation")
    # Load data for demo
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    # Feature engineering
    df['RoomsPerPop'] = df['AveRooms'] / (df['Population'] + 1e-5)
    df['BedroomRatio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
    X = df.drop('Price', axis=1)
    y = df['Price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)
    # Save model/scaler
    joblib.dump(model, 'updated_house_model.pkl')
    joblib.dump(scaler, 'updated_house_scaler.pkl')
    # User input
    st.subheader("Enter Features")
    med_inc = st.slider("Median Income", float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()))
    house_age = st.slider("House Age", float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()))
    ave_rooms = st.slider("Average Rooms", float(df['AveRooms'].min()), float(df['AveRooms'].max()), float(df['AveRooms'].mean()))
    ave_bedrms = st.slider("Average Bedrooms", float(df['AveBedrms'].min()), float(df['AveBedrms'].max()), float(df['AveBedrms'].mean()))
    population = st.slider("Population", float(df['Population'].min()), float(df['Population'].max()), float(df['Population'].mean()))
    ave_occup = st.slider("Average Occupancy", float(df['AveOccup'].min()), float(df['AveOccup'].max()), float(df['AveOccup'].mean()))
    latitude = st.slider("Latitude", float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].mean()))
    longitude = st.slider("Longitude", float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].mean()))
    # Engineered features
    rooms_per_pop = ave_rooms / (population + 1e-5)
    bedroom_ratio = ave_bedrms / (ave_rooms + 1e-5)
    # Predict
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, rooms_per_pop, bedroom_ratio]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.subheader(f"Predicted Price: ${prediction[0]:.2f}")
    # Evaluation on test data
    y_test = np.random.sample(100) * 5  # Simulated for app demo
    y_pred_test = model.predict(X_scaled[:100])  # Simulated
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    st.subheader("Model Evaluation")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R2: {r2:.2f}")

if __name__ == "__main__":
    updated_house_price_app()