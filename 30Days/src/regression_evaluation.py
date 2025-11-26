# Complex demo: Load California Housing from Day 61, evaluate linear regression with MSE, RMSE, MAE, R2, residuals, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def advanced_regression_evaluation_demo():
    # Load and prep from Day 61
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    # Feature engineering from Day 61
    df['RoomsPerPop'] = df['AveRooms'] / (df['Population'] + 1e-5)
    df['BedroomRatio'] = df['AveBedrms'] / (df['AveRooms'] + 1e-5)
    X = df.drop('Price', axis=1)
    y = df['Price']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    # Fit model from Day 61
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    # Residuals analysis
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.annotate('Mean Residual: {:.2f}'.format(np.mean(residuals)), xy=(0.05, 0.95), xycoords='axes fraction')
    plt.show()
    # Prediction vs True
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.annotate(f'R2: {r2:.2f}', xy=(y_test.mean(), y_pred.mean()), xytext=(y_test.mean() + 0.5, y_pred.mean() + 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model.coef_, y=X.columns)
    plt.title('Feature Importance')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    plt.annotate('Key Feature', xy=(model.coef_.max(), np.argmax(model.coef_)), xytext=(model.coef_.max() + 0.1, np.argmax(model.coef_) + 0.5), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

advanced_regression_evaluation_demo()