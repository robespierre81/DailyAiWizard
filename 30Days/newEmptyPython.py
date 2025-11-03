st.header("Iris Flower Classification with Full Pipeline")
# Load data for demo
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
# Feature engineering
df['Petal_Ratio'] = df['petal length (cm)'] / (df['petal width (cm)'] + 1e-5)
X = df.drop('Species', axis=1)
y = df['Species']
# Full pipeline
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
# CV scores
cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring='accuracy')
st.subheader("Cross-Validation Scores")
st.write(f"Average CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
# User input
st.subheader("Enter Iris Features")
sepal_length = st.slider("Sepal Length (cm)", ...
# Compute engineered feature
petal_ratio = petal_length / (petal_width + 1e-5)
# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width, petal_ratio]])
prediction = best_pipeline.predict(input_data)
species = iris.target_names[prediction[0]]
st.subheader(f"Predicted Species: {species}")
# Evaluation
y_pred = best_pipeline.predict(X)
accuracy = accuracy_score(y, y_pred)
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")