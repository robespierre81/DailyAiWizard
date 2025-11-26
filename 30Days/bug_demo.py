# day86_debugging_demo.py
# Run this live — it contains 6 REAL bugs that crash + fixes in seconds

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gc

# Load models (from Day 80 & 82)
@st.cache_resource
def load_models():
    img = tf.keras.models.load_model("image_model.h5")
    sent = tf.keras.models.load_model("sentiment_model.h5")
    return img, sent

img_model, sent_model = load_models()

st.title("Day 86 – 6 Bugs That Break Everyone (and How to Fix Them)")

tab1, tab2 = st.tabs(["Image Classifier", "Sentiment Analyzer"])

with tab1:
    st.header("Bug Demo: Image Classifier")
    uploaded = st.file_uploader("Upload any image", type=['png','jpg','jpeg'])

    if uploaded:
        img = Image.open(uploaded)
        
        # BUG 1: No resize → memory explosion on big images
        # BUG 2: Wrong shape → (H,W,C) instead of (1,32,32,3)
        # BUG 3: float64 → 10× slower
        arr = np.array(img) / 255.0
        
        # Uncomment ONE line at a time to show crash → fix
        # arr = arr.astype("float64")           # ← BUG 3: float64
        # arr = arr.reshape(1,32,32,3)          # ← FIX shape
        arr = img.resize((32,32))               # ← FIX resize early
        arr = np.array(arr) / 255.0
        arr = arr.reshape(1,32,32,3).astype("float32")  # ← FULL FIX
        
        pred = img_model.predict(arr)[0]
        classes = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
        st.success(f"Prediction: **{classes[np.argmax(pred)]}** – {np.max(pred):.1%}")
        
        # BUG 4: Memory leak — never delete big images
        del img, arr
        gc.collect()  # ← FIX memory leak

with tab2:
    st.header("Bug Demo: Sentiment Analyzer")
    review = st.text_area("Write a review")
    
    if st.button("Analyze"):
        if review:
            # BUG 5: No padding → wrong shape crash
            # BUG 6: Using sigmoid model with wrong preprocessing
            seq = [hash(w) % 10000 for w in review.lower().split()[:200]]
            padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=200)
            
            pred = float(sent_model.predict(padded)[0][0])
            sentiment = "POSITIVE" if pred > 0.5 else "NEGATIVE"
            
            st.metric("Sentiment", sentiment, f"{pred:.1%} confidence")
            if pred > 0.9: st.balloons()
            if pred < 0.1: st.snow()