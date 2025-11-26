import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

def updated_app_iris_training():
    st.header("Iris Classification: Full Training Pipeline")

    model_path = 'best_iris_model.h5'
    scaler_path = 'iris_training_scaler.pkl'

    # Load iris dataset ONCE, always available
    iris = load_iris()

    # Check if BOTH model and scaler exist
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.info("Training model and scaler from scratch...")
        X, y = iris.data, iris.target
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5),
            tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
        ]

        with st.spinner("Training neural network..."):
            model.fit(X_scaled, y, epochs=100, callbacks=callbacks, verbose=0)

        # Save scaler
        joblib.dump(scaler, scaler_path)
        st.success("Model and scaler trained and saved!")

    # Load model and scaler (now guaranteed to exist)
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    st.subheader("Enter Iris Features")
    col1, col2 = st.columns(2)
    with col1:
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.8, key="sl")
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.0, key="sw")
    with col2:
        pl = st.slider("Petal Length", 1.0, 7.0, 4.0, key="pl")
        pw = st.slider("Petal Width", 0.1, 2.5, 1.3, key="pw")

    # Prepare input
    input_data = np.array([[sl, sw, pl, pw]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    species = iris.target_names[predicted_class]

    st.subheader("Prediction")
    st.success(f"**Predicted: {species.capitalize()}**")

    # Optional: Show probabilities
    probs = prediction[0]
    st.write("**Class Probabilities:**")
    for name, prob in zip(iris.target_names, probs):
        st.write(f"- **{name.capitalize()}**: {prob:.3f}")

if __name__ == "__main__":
    updated_app_iris_training()