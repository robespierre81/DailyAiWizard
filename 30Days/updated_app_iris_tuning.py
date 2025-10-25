# Demo: Updated flower classification component for AI Insight Hub app using Streamlit and tuned Random Forest
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

def updated_app_iris_tuning():
    st.header("Iris Flower Classification with Tuned Random Forest")
    # Load data for demo
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 3]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    # Save model/scaler
    joblib.dump(best_model, 'iris_tuned_model.pkl')
    joblib.dump(scaler, 'iris_tuned_scaler.pkl')
    # User input
    st.subheader("Enter Iris Features")
    sepal_length = st.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    # Compute engineered feature
    petal_ratio = petal_length / (petal_width + 1e-5)
    # Predict
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width, petal_ratio]])
    input_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_scaled)
    species = iris.target_names[prediction[0]]
    st.subheader(f"Predicted Species: {species}")
    # Evaluation
    y_pred = best_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    updated_app_iris_tuning()