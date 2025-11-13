import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os

def updated_app_cnn_digit():
    st.header("Draw a Digit – CNN Reads It!")
    
    if not os.path.exists('cnn_digit_model.h5'):
        st.info("Training CNN for the first time...")
        (X_train, _), _ = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, np.arange(len(X_train)) % 10, epochs=3, verbose=0)
        model.save('cnn_digit_model.h5')
        st.success("CNN trained!")
    
    model = tf.keras.models.load_model('cnn_digit_model.h5')
    
    canvas_result = st.canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=280, width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit]
        
        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width=100)
        with col2:
            st.success(f"**Digit: {digit}**\nConfidence: {confidence:.1%}")

if __name__ == "__main__":
    updated_app_cnn_digit()