# Demo: Updated flower clustering component for AI Insight Hub app using Streamlit and K-Means
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def updated_iris_kmeans_app():
    st.header("Iris Flower Clustering with K-Means")
    # Load data for demo
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.values
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)
    labels = model.labels_
    # Save model/scaler
    joblib.dump(model, 'iris_kmeans_model.pkl')
    joblib.dump(scaler, 'iris_kmeans_scaler.pkl')
    # User input
    st.subheader("Enter Iris Features")
    sepal_length = st.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
    # Compute engineered feature
    petal_ratio = petal_length / (petal_width + 1e-5)
    # Predict cluster
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width, petal_ratio]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    st.subheader(f"Predicted Cluster: Cluster {cluster}")
    # Evaluation
    silhouette = silhouette_score(X_scaled, labels)
    st.subheader("Model Silhouette Score")
    st.write(f"Silhouette Score: {silhouette:.2f}")
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
    ax.scatter(input_scaled[0, 0], input_scaled[0, 1], c='black', marker='*', s=300, label='Input')
    ax.set_title('K-Means: Iris Clustering with Input')
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Sepal Width (scaled)')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    updated_iris_kmeans_app()