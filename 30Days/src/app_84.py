# app.py – The Ultimate Day 84 Demo App
# Works on: Streamlit Share, Hugging Face, FastAPI, Flask, TF Serving, Cloud, TFLite

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

# -------------------------------
# 1. Load both saved models (Day 83)
# -------------------------------
@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("image_model.h5")   # Day 80
    sent_model = tf.keras.models.load_model("sentiment_model.h5")  # Day 82
    return img_model, sent_model

img_model, sent_model = load_models()

# CIFAR-10 class names
IMG_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Day 84 – All Deployments!", page_icon="rocket", layout="wide")
st.title("Day 84: 8 Ways to Deploy – ONE App!")
st.markdown("**Image Classification + Sentiment Analysis** – Live on every platform!")

tab1, tab2 = st.tabs(["Image Classifier", "Sentiment Analyzer"])

# -------------------------------
# TAB 1: Image Classification
# -------------------------------
with tab1:
    st.header("Upload an image (CIFAR-10 model)")
    uploaded = st.file_uploader("Choose PNG/JPG...", type=['png','jpg','jpeg'], key="img")
    
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        img_resized = img.resize((32,32))
        arr = np.array(img_resized) / 255.0
        pred = img_model.predict(arr.reshape(1,32,32,3), verbose=0)[0]
        idx = np.argmax(pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_column_width=True)
        with col2:
            st.success(f"**{IMG_CLASSES[idx].upper()}**")
            st.bar_chart(dict(zip(IMG_CLASSES, pred)))
            if pred[idx] > 0.95:
                st.balloons()

# -------------------------------
# TAB 2: Sentiment Analysis
# -------------------------------
with tab2:
    st.header("Write a movie review")
    review = st.text_area("Your review:", height=150, placeholder="I loved this movie because...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if review.strip():
            # Simple fake tokenizer (good enough for demo)
            seq = [abs(hash(w)) % 10000 for w in review.lower().split()[:200]]
            padded = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=200)
            pred = float(sent_model.predict(padded, verbose=0)[0][0])
            sentiment = "POSITIVE" if pred > 0.5 else "NEGATIVE"
            
            st.metric("Sentiment", sentiment, f"{pred:.1%} confidence")
            if pred > 0.9: st.balloons()
            if pred < 0.1: st.snow()
        else:
            st.warning("Write something first!")

# -------------------------------
# Sidebar – Credits & Links
# -------------------------------
st.sidebar.image("https://dailyaiwizard.com/logo.png", width=200)
st.sidebar.success("Day 84 Live Demo!")
st.sidebar.markdown("""
- Image model: Day 80 CIFAR-10  
- Sentiment model: Day 82 IMDB  
- Saved with Day 83  
- Deployed with Day 84  
""")
st.sidebar.markdown("Support us → [buymeacoffee.com/dailyaiwizard](https://buymeacoffee.com/dailyaiwizard)")