# updated_app_iris_full.py
# Demo: Updated flower classification component for AI Insight Hub app using Streamlit and full pipeline
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score  # <-- ADDED
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

def updated_app_iris_full():
    st.set_page_config(page_title="AI Insight Hub - Iris Classifier", layout="wide")
    st.header("Iris Flower Classification with Full Pipeline")

    # -------------------------------------------------
    # 1. Load data
    # -------------------------------------------------
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target

    # -------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------
    df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
    X = df.drop('Species', axis=1)
    y = df['Species']

    # -------------------------------------------------
    # 3. Full pipeline with PCA and Random Forest
    # -------------------------------------------------
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [3]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_

    # Save pipeline
    joblib.dump(best_pipeline, 'iris_full_pipeline.pkl')

    # -------------------------------------------------
    # 4. Cross-validation scores
    # -------------------------------------------------
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='accuracy')  # <-- Now works!
    st.subheader("Cross-Validation Scores")
    st.write(f"Average CV Score: {cv_scores.mean():.3f} (± {cv_scores.std() * 2:.3f})")

    # -------------------------------------------------
    # 5. User input sliders
    # -------------------------------------------------
    st.subheader("Enter Iris Features")
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
        sepal_width = st.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))

    with col2:
        petal_length = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
        petal_width = st.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

    # Compute engineered feature
    petal_ratio = petal_length / (petal_width + 1e-5)

    # -------------------------------------------------
    # 6. Predict
    # -------------------------------------------------
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width, petal_ratio]])
    prediction = best_pipeline.predict(input_data)
    species = iris.target_names[prediction[0]]

    st.subheader("Prediction")
    st.success(f"**Predicted Species: {species.capitalize()}**")

    # -------------------------------------------------
    # 7. Optional: Visualize PCA space
    # -------------------------------------------------
    if st.checkbox("Show Classification in PCA Space"):
        X_scaled = best_pipeline.named_steps['scaler'].transform(X)
        X_pca = best_pipeline.named_steps['pca'].transform(X_scaled)
        y_pred = best_pipeline.predict(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k')
        ax.scatter(
            best_pipeline.named_steps['pca'].transform(
                best_pipeline.named_steps['scaler'].transform(input_data)
            )[0],
            c='red', marker='*', s=300, label='Your Input'
        )
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Iris Classification in PCA Space')
        ax.legend()
        st.pyplot(fig)

# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    updated_app_iris_full()