# house_price_demo.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    root_mean_squared_error,   # <-- NEW import
    r2_score,
    mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns

def house_price_demo():
    # -------------------------------------------------
    # 1. Load data (make sure housing.csv is in the folder)
    # -------------------------------------------------
    df = pd.read_csv('housing.csv')
    df = df.dropna()                     # drop rows with missing total_bedrooms

    # -------------------------------------------------
    # 2. Feature engineering (same as in the video)
    # -------------------------------------------------
    df['rooms_per_house'] = df['total_rooms'] / (df['households'] + 1e-5)
    df['bedrooms_per_room'] = df['total_bedrooms'] / (df['total_rooms'] + 1e-5)

    X = df.drop(['median_house_value', 'ocean_proximity'], axis=1)
    y = df['median_house_value']

    # -------------------------------------------------
    # 3. Scale features
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------------------------------
    # 4. Train / test split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # -------------------------------------------------
    # 5. Model (tuned parameters from Day 69)
    # -------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # 6. Cross‑validation (R²)
    # -------------------------------------------------
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"Cross‑Validation R² Scores: {cv_scores}")
    print(f"Average CV R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std()*2:.2f})")

    # -------------------------------------------------
    # 7. Evaluation on the hold‑out test set
    # -------------------------------------------------
    rmse = root_mean_squared_error(y_test, y_pred)   # <-- NEW line
    r2   = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:,.0f}")
    print(f"Test R²: {r2:.2f}")

    # -------------------------------------------------
    # 8. Visualise predictions (actual vs predicted)
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='steelblue', alpha=0.5, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual House Price ($)')
    plt.ylabel('Predicted House Price ($)')
    plt.title('Random Forest – House Price Prediction')
    plt.annotate(f'RMSE = {rmse:,.0f}\nR² = {r2:.2f}',
                 xy=(0.05, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    plt.show()

    # -------------------------------------------------
    # 9. Feature importance (optional but nice)
    # -------------------------------------------------
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values(ascending=False).plot(kind='barh', figsize=(8, 5))
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Run the demo
# -------------------------------------------------
if __name__ == "__main__":
    house_price_demo()