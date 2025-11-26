# Complex demo: Load Iris, preprocess, apply k-fold CV to Random Forest, evaluate, visualize with Matplotlib/Seaborn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def iris_cv_demo():
    # Load and prep
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    # Feature engineering
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Set up Random Forest model (using tuned parameters from Day 69)
    model = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=42)
    # K-fold cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"K-Fold CV Scores: {cv_scores}")
    print(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    print(f"Stratified K-Fold CV Scores: {skf_scores}")
    print(f"Average Stratified CV Score: {skf_scores.mean():.2f} (+/- {skf_scores.std() * 2:.2f})")
    # Visualize CV scores
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 6), cv_scores, marker='o', label='K-Fold CV')
    plt.plot(range(1, 6), skf_scores, marker='s', label='Stratified K-Fold CV')
    plt.title('Cross-Validation Scores for Random Forest')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.annotate(f'Avg K-Fold: {cv_scores.mean():.2f}', xy=(2, cv_scores.mean()), xytext=(3, cv_scores.mean() + 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    # Train final model for confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Precision: {precision:.2f}")
    print(f"Test Recall: {recall:.2f}")
    # Visualize confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix for Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.annotate(f'Accuracy: {accuracy:.2f}', xy=(1, 1), xytext=(2, 2), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()

iris_cv_demo()