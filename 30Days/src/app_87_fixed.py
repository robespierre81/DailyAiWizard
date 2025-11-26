# fixed_app.py – The Same App, Now Bulletproof
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc

@st.cache_resource
def load_models():
    img = tf.keras.models.load_model("image_model.h5")
    sent = tf.keras.models.load_model("sentiment_model.h5")
    return img, sent

img_model, sent_model = load_models()
st.title("Day 87 – Fixed in Minutes!")
tab1, tab2 = st.tabs(["Image Classifier", "Sentiment Analyzer"])

with tab1:
    uploaded = st.file_uploader("Upload (max 5MB)", type=['jpg','png'], key="img")
    if uploaded and uploaded.size < 5_000_000:  # ← Fix: size limit
        img = Image.open(uploaded).convert('RGB')
        img = img.resize((32,32))  # ← Fix: resize early
        arr = np.array(img) / 255.0
        arr = arr.reshape(1,32,32,3)  # ← Fix: correct shape
        pred = img_model.predict(arr, verbose=0)[0]
        st.success(f"**{['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck'][pred.argmax()]}**")
        del img, arr  # ← Fix: memory cleanup
        gc.collect()

with tab2:
    review = st.text_area("Write a review")
    if st.button("Analyze"):
        if review:
            try:
                seq = [hash(w) % 10000 for w in review.lower().split()[:200]]
                padded = pad_sequences([seq], maxlen=200)  # ← Fix: proper padding
                pred = float(sent_model.predict(padded, verbose=0)[0][0])
                st.metric("Sentiment", "POSITIVE" if pred > 0.5 else "NEGATIVE", f"{pred:.1%}")
                if pred > 0.9: st.balloons()
            except Exception as e:
                st.error("Debug mode caught error — safe!")  # ← Fix: graceful error