import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved sentiment model from Day 82/83
@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model('sentiment_model.h5')

model = load_sentiment_model()

# Simple tokenizer (same as training)
def preprocess_text(text):
    # Very simple tokenizer for demo
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    return padded

st.title("Day 84 Live Demo – Instant Deployment!")
st.write("This app uses the sentiment model you saved in Day 83 — loads instantly!")
review = st.text_area("Write a movie review:", height=150)

if review:
    processed = preprocess_text(review)
    pred = model.predict(processed)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    
    st.success(f"AI says: **{sentiment}**")
    st.progress(float(pred) if pred > 0.5 else 1 - float(pred))
    st.write(f"Confidence: {max(pred, 1-pred):.1%}")