import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def updated_app_iris_loss():
    st.header("Iris Classification: Loss Function Comparison")
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    losses = ['sparse_categorical_crossentropy', 'categorical_crossentropy']
    models = {}
    accuracies = {}
    
    for loss in losses:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        if loss == 'categorical_crossentropy':
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            y_onehot = encoder.fit_transform(y.reshape(-1, 1))
            model.fit(X_scaled, y_onehot, epochs=50, verbose=0)
            test_loss, test_acc = model.evaluate(X_scaled, y_onehot, verbose=0)
        else:
            model.fit(X_scaled, y, epochs=50, verbose=0)
            test_loss, test_acc = model.evaluate(X_scaled, y, verbose=0)
        
        models[loss] = model
        accuracies[loss] = test_acc
    
    best_loss = max(accuracies, key=accuracies.get)
    best_model = models[best_loss]
    joblib.dump(best_model, f'iris_loss_{best_loss}.h5')
    joblib.dump(scaler, 'iris_loss_scaler.pkl')
    
    st.subheader("Loss Function Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sparse CE", f"{accuracies['sparse_categorical_crossentropy']:.3f}")
    with col2:
        st.metric("Categorical CE", f"{accuracies['categorical_crossentropy']:.3f}")
    
    st.info(f"**Best: {best_loss} with {accuracies[best_loss]:.3f} accuracy**")
    
    st.subheader("Enter Iris Features")
    col1, col2 = st.columns(2)
    with col1:
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    with col2:
        pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
        pw = st.slider("Petal Width", 0.1, 2.5, 1.3)
    
    input_data = scaler.transform([[sl, sw, pl, pw]])
    prediction = best_model.predict(input_data)
    species = iris.target_names[np.argmax(prediction)]
    
    st.subheader("Prediction")
    st.success(f"**Predicted: {species.capitalize()}**")
    
    selected_loss = st.selectbox("Try Different Loss", losses)
    if selected_loss != best_loss:
        test_pred = models[selected_loss].predict(input_data)
        sel_species = iris.target_names[np.argmax(test_pred)]
        st.info(f"With {selected_loss}: {sel_species}")

if __name__ == "__main__":
    updated_app_iris_loss()