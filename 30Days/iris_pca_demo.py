# Complex demo: Load Iris, preprocess, fit PCA, reduce, evaluate with variance and silhouette, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def iris_pca_demo():
    # Load and prep
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
    # Fit PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # Evaluate explained variance
    explained_variance = pca.explained_variance_ratio_
    total_variance = sum(explained_variance)
    print("Explained Variance Ratio:", explained_variance)
    print("Total Explained Variance:", total_variance)
    # Silhouette score for evaluation (using KMeans on reduced data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_pca)
    silhouette = silhouette_score(X_pca, kmeans.labels_)
    print(f"Silhouette Score: {silhouette:.2f}")
    # Visualize reduced dimensions
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('PCA of Iris dataset')
    plt.colorbar(scatter, ticks=[0, 1, 2], format=plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)]))
    plt.annotate(f'Total Variance: {total_variance:.2f}', xy=(0, 0), xytext=(1, 1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Visualize silhouette scores for different k
    silhouette_scores = []
    for k in range(2, 6):
        kmeans_k = KMeans(n_clusters=k, random_state=42)
        kmeans_k.fit(X_pca)
        score = silhouette_score(X_pca, kmeans_k.labels_)
        silhouette_scores.append(score)
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 6), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.annotate(f'Best k: 3', xy=(3, silhouette_scores[1]), xytext=(3.5, silhouette_scores[1] + 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

iris_pca_demo()