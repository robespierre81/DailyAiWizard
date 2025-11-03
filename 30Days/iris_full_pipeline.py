# Complex demo: Full Iris classification pipeline with preprocessing, PCA, tuning, CV, Random Forest, evaluation, visualization
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def iris_full_pipeline_demo():
    # Load and prep
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Full pipeline with PCA and tuning
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 3]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    # Cross-validation on best pipeline
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"CV Scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    # Visualize predictions in PCA space
    X_pca = best_pipeline.named_steps['pca'].transform(best_pipeline.named_steps['scaler'].transform(X_test))
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.title('Iris Classification in PCA Space')
    plt.annotate(f'Accuracy: {accuracy:.2f}', xy=(0, 0), xytext=(1, 1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix for Full Pipeline')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

iris_full_pipeline_demo()