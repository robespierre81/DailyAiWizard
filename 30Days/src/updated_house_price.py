# Demo: Updated house price prediction component for AI Insight Hub app using Streamlit and Random Forest
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def updated_app_house_price():
    st.header("House Price Prediction with Random Forest")
    # Load data for demo
    df = pd.read_csv('housing.csv')
    df = df.dropna()
    # Feature engineering
    df['rooms_per_house'] = df['total_rooms'] / (df['households'] + 1e-5)
    df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + 1e-5)
    X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
    y = df['median_house_value']
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_scaled, y)
    # Save model/scaler
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'house_price_scaler.pkl')
    # User input
    st.subheader("Enter House Features")
    total_rooms = st.slider("Total Rooms", float(df['total_rooms'].min()), float(df['total_rooms'].max()), float(df['total_rooms'].mean()))
    total_bedrooms = st.slider("Total Bedrooms", float(df['total_bedrooms'].min()), float(df['total_bedrooms'].max()), float(df['total_bedrooms'].mean()))
    population = st.slider("Population", float(df['population'].min()), float(df['population'].max()), float(df['population'].mean()))
    households = st.slider("Households", float(df['households'].min()), float(df['households'].max()), float(df['households'].mean()))
    median_income = st.slider("Median Income", float(df['median_income'].min()), float(df['median_income'].max()), float(df['median_income'].mean()))
    # Compute engineered features
    rooms_per_house = total_rooms / (households + 1e-5)
    bedrooms_per_room = total_bedrooms / (total_rooms + 1e-5)
    # Predict
    input_data = np.array([[total_rooms, total_bedrooms, population, households, median_income, rooms_per_house, bedrooms_per_room]])
    input