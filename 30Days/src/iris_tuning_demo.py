# Complex demo: Load Iris from previous days, preprocess, tune hyperparameters for Random Forest using GridSearchCV, evaluate, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def iris_tuning_demo():
    # Load and prep from previous days
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5],
        'criterion': ['gini', 'entropy']
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix for Tuned Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.annotate(f'Accuracy: {accuracy:.2f}', xy=(1, 1), xytext=(2, 2), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
    plt.title('Tuned Random Forest: Iris Predictions')
    plt.xlabel('Sepal Length (scaled)')
    plt.ylabel('Sepal Width (scaled)')
    plt.annotate(f'Accuracy: {accuracy:.2f}', xy=(0, 0), xytext=(1, 1), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

iris_tuning_demo()