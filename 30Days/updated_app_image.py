import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os

def updated_app_image():
    st.header("Upload an Image – AI Classifies It!")
    
    if not os.path.exists('image_model.h5'):
        st.info("Training model for the first time...")
        (X_train, _), _ = tf.keras.datasets.cifar10.load_data()
        X_train = X_train / 255.0
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, np.arange(len(X_train)) % 10, epochs=5, verbose=0)
        model.save('image_model.h5')
        st.success("Model trained!")
    
    model = tf.keras.models.load_model('image_model.h5')
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((32, 32))
        img_array = np.array(img_resized) / 255.0
        img_array = img_array.reshape(1, 32, 32, 3)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, width=200)
        with col2:
            st.success(f"**Class: {class_names[predicted_class].capitalize()}**\nConfidence: {confidence:.1%}")

if __name__ == "__main__":
    updated_app_image()