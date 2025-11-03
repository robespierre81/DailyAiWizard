# updatedured_app_house_price.py
# Demo: Updated house price prediction component for AI Insight Hub app using Streamlit and Random Forest
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

def updated_app_house_price():
    st.set_page_config(page_title="AI Insight Hub - House Price Predictor", layout="wide")
    st.header("House Price Prediction with Random Forest")

    # -------------------------------------------------
    # 1. Load and preprocess data
    # -------------------------------------------------
    try:
        df = pd.read_csv('housing.csv')
    except FileNotFoundError:
        st.error("`housing.csv` not found! Please place it in the same folder.")
        st.stop()

    df = df.dropna()

    # -------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------
    df['rooms_per_house'] = df['total_rooms'] / (df['households'] + 1e-5)
    df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + 1e-5)

    # Keep ALL original + engineered features
    feature_cols = [
        'longitude', 'latitude', 'housing_median_age',
        'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income',
        'rooms_per_house', 'bedrooms_per_room'
    ]
    X = df[feature_cols]
    y = df['median_house_value']

    # -------------------------------------------------
    # 3. Scale features
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------
    # 4. Train model
    # -------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'house_price_scaler.pkl')

    # -------------------------------------------------
    # 5. Model evaluation
    # -------------------------------------------------
    y_pred = model.predict(X_scaled)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"${rmse:,.0f}")
    with col2:
        st.metric("R² Score", f"{r2:.3f}")

    # -------------------------------------------------
    # 6. User input sliders
    # -------------------------------------------------
    st.subheader("Enter House Features")
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.slider("Longitude", float(df['longitude'].min()), float(df['longitude'].max()), float(df['longitude'].mean()))
        latitude = st.slider("Latitude", float(df['latitude'].min()), float(df['latitude'].max()), float(df['latitude'].mean()))
        housing_median_age = st.slider("Housing Median Age", float(df['housing_median_age'].min()), float(df['housing_median_age'].max()), float(df['housing_median_age'].mean()))
        total_rooms = st.slider("Total Rooms", float(df['total_rooms'].min()), float(df['total_rooms'].max()), float(df['total_rooms'].mean()))
        total_bedrooms = st.slider("Total Bedrooms", float(df['total_bedrooms'].min()), float(df['total_bedrooms'].max()), float(df['total_bedrooms'].mean()))

    with col2:
        population = st.slider("Population", float(df['population'].min()), float(df['population'].max()), float(df['population'].mean()))
        households = st.slider("Households", float(df['households'].min()), float(df['households'].max()), float(df['households'].mean()))
        median_income = st.slider("Median Income (x$10k)", float(df['median_income'].min()), float(df['median_income'].max()), float(df['median_income'].mean()))

    # -------------------------------------------------
    # 7. Compute engineered features
    # -------------------------------------------------
    rooms_per_house = total_rooms / (households + 1e-5)
    bedrooms_per_room = total_bedrooms / (total_rooms + 1e-5)

    # -------------------------------------------------
    # 8. Prepare input with ALL 10 features
    # -------------------------------------------------
    input_data = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        rooms_per_house,
        bedrooms_per_room
    ]])

    input_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_scaled)[0]

    st.subheader("Prediction")
    st.success(f"**Predicted House Price: ${predicted_price:,.0f}**")

    # -------------------------------------------------
    # 9. Optional: Visualize actual vs predicted
    # -------------------------------------------------
    if st.checkbox("Show Prediction vs Actual (Sample)"):
        sample_pred = model.predict(X_scaled[:100])
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y[:100], sample_pred, alpha=0.6, color='teal')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Model Prediction vs Actual (Sample)")
        st.pyplot(fig)

# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    updated_app_house_price()