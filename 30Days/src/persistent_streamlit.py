import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.set_page_config(page_title="Day 83 – Immortal AI", layout="centered")
st.title("Day 83 Live Demo – Immortal Sentiment AI")

@st.cache_resource
def load_model():
    if not os.path.exists("sentiment_model.h5"):
        st.error("Model not found! Run save_load_sentiment_demo.py first.")
        st.stop()
    return tf.keras.models.load_model("sentiment_model.h5")

model = load_model()
st.success("Model loaded instantly from Day 83!")

review = st.text_area("Write a movie review:", height=150, placeholder="I loved this film...")

if st.button("Analyze Sentiment"):
    if review:
        # Very simple preprocessing for demo
        seq = tf.keras.preprocessing.text.text_to_word_sequence(review.lower())
        # Fake tokenization (just for demo)
        seq = [[min(hash(w) % 10000, 9999) for w in seq[:200]]]
        padded = pad_sequences(seq, maxlen=200)
        
        pred = model.predict(padded)[0][0]
        sentiment = "Positive" if pred > 0.5 else "Negative"
        
        st.metric("Sentiment", sentiment, f"{pred:.1%} confidence")
        if pred > 0.5:
            st.balloons()
        else:
            st.snow()
    else:
        st.warning("Write something first!")