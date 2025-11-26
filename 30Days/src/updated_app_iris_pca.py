# Demo: Updated flower reduction component for AI Insight Hub app using Streamlit and PCA
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def updated_app_iris_pca():
    st.header("Iris PCA Dimensionality Reduction")
    # Load data for demo
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # User input for n_components
    n_components = st.slider("Number of PCA Components", 1, X.shape[1], 2)
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    # Save model/scaler
    joblib.dump(pca, 'iris_pca_model.pkl')
    joblib.dump(scaler, 'iris_pca_scaler.pkl')
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = sum(explained_variance)
    st.subheader("Explained Variance Ratio")
    st.write(explained_variance)
    st.write(f"Total Explained Variance: {total_variance:.2f}")
    # Silhouette score for evaluation (using KMeans on reduced data)
    if n_components >= 2:
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X_pca)
        silhouette = silhouette_score(X_pca, kmeans.labels_)
        st.subheader("Silhouette Score for PCA-reduced Data")
        st.write(f"Silhouette Score: {silhouette:.2f}")
        # Visualize reduced dimensions
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Species'], cmap='viridis', edgecolor='k', s=50)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second principal component')
        ax.set_title('PCA of Iris dataset')
        fig.colorbar(scatter, ax=ax, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)]))
        ax.annotate(f'Total Variance: {total_variance:.2f}', xy=(0, 0), xytext=(1, 1), arrowprops=dict(facecolor='black', shrink=0.05))
        st.pyplot(fig)
    else:
        st.write("Visualization available for 2 or more components.")

if __name__ == "__main__":
    updated_app_iris_pca()