import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def updated_app_iris_nn():
    st.header("Iris Classification with Neural Network")
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=50, verbose=0)
    
    joblib.dump(model, 'iris_nn_model.pkl')
    joblib.dump(scaler, 'iris_nn_scaler.pkl')
    
    st.subheader("Enter Iris Features")
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
    sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
    pw = st.slider("Petal Width", 0.1, 2.5, 1.3)
    
    input_data = scaler.transform([[sl, sw, pl, pw]])
    pred = model.predict(input_data)
    species = iris.target_names[np.argmax(pred)]
    st.success(f"**Predicted: {species.capitalize()}**")

if __name__ == "__main__":
    updated_app_iris_nn()