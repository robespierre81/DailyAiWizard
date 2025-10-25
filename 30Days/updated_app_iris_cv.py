# Demo: Updated flower classification component for AI Insight Hub app using Streamlit and cross-validation
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def updated_app_iris_cv():
    st.header("Iris Flower Classification with Cross-Validation")
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
    # Train model with tuned parameters from Day 69
    model = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=42)
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    # Save model/scaler
    model.fit(X_scaled, y)
    joblib.dump(model, 'iris_cv_model.pkl')
    joblib.dump(scaler, 'iris_cv_scaler.pkl')
    # Display CV results
    st.subheader("Cross-Validation Scores")
    st.write(f"K-Fold CV Scores: {cv_scores}")
    st.write(f"Average K-Fold CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    st.write(f"Stratified K-Fold CV Scores: {skf_scores}")
    st.write(f"Average Stratified K-Fold CV Score: {skf_scores.mean():.2f} (+/- {skf_scores.std() * 2:.2f})")
    # Visualize CV scores
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, 6), cv_scores, marker='o', label='K-Fold CV')
    ax.plot(range(1, 6), skf_scores, marker='s', label='Stratified K-Fold CV')
    ax.set_title('Cross-Validation Scores for Random Forest')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.annotate(f'Avg K-Fold: {cv_scores.mean():.2f}', xy=(2, cv_scores.mean()), xytext=(3, cv_scores.mean() + 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
    st.pyplot(fig)
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
    updated_app_iris_cv()