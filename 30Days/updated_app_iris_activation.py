import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def updated_app_iris_activation():
    st.header("Iris Classification: Activation Function Comparison")
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models with different activations
    activations = ['relu', 'sigmoid', 'tanh']
    models = {}
    accuracies = {}
    
    for act in activations:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation=act, input_shape=(4,)),
            tf.keras.layers.Dense(16, activation=act),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled, y, epochs=50, verbose=0)
        
        test_loss, test_acc = model.evaluate(X_scaled, y, verbose=0)
        models[act] = model
        accuracies[act] = test_acc
    
    # Save best model
    best_act = max(accuracies, key=accuracies.get)
    best_model = models[best_act]
    joblib.dump(best_model, f'iris_activation_{best_act}.h5')
    joblib.dump(scaler, 'iris_activation_scaler.pkl')
    
    # Display results
    st.subheader("Activation Function Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ReLU", f"{accuracies['relu']:.3f}")
    with col2:
        st.metric("Sigmoid", f"{accuracies['sigmoid']:.3f}")
    with col3:
        st.metric("Tanh", f"{accuracies['tanh']:.3f}")
    
    st.info(f"**Best: {best_act.upper()} with {accuracies[best_act]:.3f} accuracy**")
    
    # User input
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
    
    # Activation selector
    selected_act = st.selectbox("Try Different Activation", activations)
    if selected_act != best_act:
        test_pred = models[selected_act].predict(input_data)
        sel_species = iris.target_names[np.argmax(test_pred)]
        st.info(f"With {selected_act}: {sel_species}")

if __name__ == "__main__":
    updated_app_iris_activation()