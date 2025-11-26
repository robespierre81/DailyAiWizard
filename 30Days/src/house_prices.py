# Complex demo: Load house price data, preprocess, engineer features, fit Random Forest regression, evaluate, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def house_price_demo():
    # Load house price data (assuming housing.csv is available)
    df = pd.read_csv('housing.csv')
    # Handle missing values
    df = df.dropna()
    # Feature engineering
    df['rooms_per_house'] = df['total_rooms'] / (df['households'] + 1e-5)
    df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + 1e-5)
    X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)  # Drop target and categorical for simplicity
    y = df['median_house_value']
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    # Fit Random Forest with tuned parameters from Day 69
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"Cross-Validation R² Scores: {cv_scores}")
    print(f"Average CV R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    # Evaluate
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R²: {r2:.2f}")
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.title('House Price Prediction with Random Forest')
    plt.annotate(f'RMSE: {rmse:.2f}, R²: {r2:.2f}', xy=(0.1, 0.9), xytext=(0.2, 0.8), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Visualize feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    plt.figure(figsize=(8, 6))
    feature_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance in Random Forest')
    plt.xlabel('Importance')
    plt.show()

house_price_demo()