import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

def updated_app_iris_optimizer():
    st.header("Iris Classification: Optimizer Comparison")
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    optimizers = {
        'adam': tf.keras.optimizers.Adam(),
        'sgd': tf.keras.optimizers.SGD(),
        'rmsprop': tf.keras.optimizers.RMSprop()
    }
    models = {}
    accuracies = {}
    
    for name, opt in optimizers.items():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled, y, epochs=50, verbose=0)
        
        test_loss, test_acc = model.evaluate(X_scaled, y, verbose=0)
        models[name] = model
        accuracies[name] = test_acc
    
    best_opt = max(accuracies, key=accuracies.get)
    best_model = models[best_opt]
    joblib.dump(best_model, f'iris_optimizer_{best_opt}.h5')
    joblib.dump(scaler, 'iris_optimizer_scaler.pkl')
    
    st.subheader("Optimizer Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Adam", f"{accuracies['adam']:.3f}")
    with col2:
        st.metric("SGD", f"{accuracies['sgd']:.3f}")
    with col3:
        st.metric("RMSprop", f"{accuracies['rmsprop']:.3f}")
    
    st.info(f"**Best: {best_opt.upper()} with {accuracies[best_opt]:.3f} accuracy**")
    
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
    
    selected_opt = st.selectbox("Try Different Optimizer", list(optimizers.keys()))
    if selected_opt != best_opt:
        test_pred = models[selected_opt].predict(input_data)
        sel_species = iris.target_names[np.argmax(test_pred)]
        st.info(f"With {selected_opt}: {sel_species}")

if __name__ == "__main__":
    updated_app_iris_optimizer()