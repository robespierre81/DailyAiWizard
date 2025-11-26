# broken_app.py – 8 REAL-WORLD BUGS INJECTED ON PURPOSE
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# BUG 1: Wrong initialization (dead layers) + sigmoid in deep net (vanishing gradients)
@st.cache_resource
def load_models():
    img = tf.keras.models.load_model("image_model.h5")
    sent = tf.keras.models.load_model("sentiment_model.h5")
    return img, sent

img_model, sent_model = load_models()

st.title("Day 87 – Broken on Purpose!")

tab1, tab2 = st.tabs(["Image", "Text"])

with tab1:
    uploaded = st.file_uploader("Upload", type=['jpg','png'])
    if uploaded:
        img = Image.open(uploaded)
        # BUG 2: No resizing → memory explosion on big images
        arr = np.array(img) / 255.0
        # BUG 3: Wrong shape → (H,W,C) instead of (1,32,32,3)
        pred = img_model.predict(arr)[0]
        st.write("Prediction:", pred.argmax())

with tab2:
    text = st.text_area("Review")
    if st.button("Predict"):
        # BUG 4: No padding → crashes on short text
        # BUG 5: Using sigmoid model with wrong preprocessing
        seq = [1, 2, 3]  # fake
        padded = tf.constant(seq)  # wrong shape
        pred = sent_model.predict(padded)  # ← WILL CRASH
        st.write(pred)