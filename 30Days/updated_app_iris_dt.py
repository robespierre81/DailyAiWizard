# Demo: Updated flower classification component for AI Insight Hub app using Streamlit and Decision Tree
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def updated_iris_dt_app():
    st.header("Iris Flower Classification with Decision Tree")
    # Load data for demo
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
    model.fit(X_scaled, y)
    # Save model/scaler
    joblib.dump(model, 'iris_dt_model.pkl')
    joblib.dump(scaler, 'iris_dt_scaler.pkl')
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
    prediction = model.predict(input_scaled)
    species = iris.target_names[prediction[0]]
    st.subheader(f"Predicted Species: {species}")
    # Evaluation
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    updated_iris_dt_app()