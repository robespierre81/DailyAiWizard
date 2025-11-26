# Complex demo: Load Iris, preprocess, fit K-Means clustering, assign clusters, evaluate, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def iris_kmeans_demo():
    # Load and prep
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.values
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Fit K-Means
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X_scaled)
    labels = model.labels_
    # Evaluate
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette:.2f}")
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title('K-Means: Iris Clustering')
    plt.xlabel('Sepal Length (scaled)')
    plt.ylabel('Sepal Width (scaled)')
    plt.annotate(f'Silhouette: {silhouette:.2f}', xy=(0, 0), xytext=(1, 1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()
    plt.show()
    # Visualize silhouette scores for k=2 to k=5
    silhouette_scores = []
    for k in range(2, 6):
        model_k = KMeans(n_clusters=k, random_state=42)
        model_k.fit(X_scaled)
        score = silhouette_score(X_scaled, model_k.labels_)
        silhouette_scores.append(score)
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 6), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

iris_kmeans_demo()