# Complex demo: Load California Housing, preprocess, feature engineer, fit linear regression, predict, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def house_price_linear_demo():
    # Load and prep
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    # Feature engineering
    df['RoomsPerPop'] = df['AveRooms'] / (df['Population'] + 1e-5)
    df['BedroomRatio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
    # Preprocessing
    X = df.drop('Price', axis=1)
    y = df['Price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}, R2: {r2:.2f}")
    print(f"Model Coefficients: {model.coef_}")
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, c='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Linear Regression: House Price Predictions')
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.annotate(f'R2: {r2:.2f}', xy=(y_test.mean(), y_pred.mean()), xytext=(y_test.mean() + 0.5, y_pred.mean() + 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.coef_, y=X.columns)
    plt.title('Feature Importance in Linear Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.annotate('Key Feature', xy=(model.coef_.max(), np.argmax(model.coef_)), xytext=(model.coef_.max() + 0.1, np.argmax(model.coef_) + 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

house_price_linear_demo()