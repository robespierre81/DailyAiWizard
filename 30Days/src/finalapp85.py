import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="My AI Web App", layout="centered")
st.title("My First AI Web App – Day 85 Launch!")

# Load models instantly
@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("image_model.h5")
    sent_model = tf.keras.models.load_model("sentiment_model.h5")
    return img_model, sent_model

img_model, sent_model = load_models()

tab1, tab2 = st.tabs(["Image Classifier", "Sentiment Analyzer"])

with tab1:
    st.header("Upload an image")
    uploaded = st.file_uploader("Choose image...", type=['png','jpg','jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        img_resized = img.resize((32,32))
        arr = np.array(img_resized) / 255.0
        pred = img_model.predict(arr.reshape(1,32,32,3))[0]
        classes = ['airplane','car','bird','cat','deer','dog','frog','horse','ship','truck']
        st.image(img, use_column_width=True)
        st.success(f"**{classes[np.argmax(pred)]}** – {np.max(pred):.1%} confident")
        if np.max(pred) > 0.95: st.balloons()

with tab2:
    st.header("Write a movie review")
    review = st.text_area("Your review:", height=150)
    if st.button("Analyze"):
        if review:
            # Simple preprocessing
            seq = [hash(w) % 10000 for w in review.lower().split()[:200]]
            padded = pad_sequences([seq], maxlen=200)
            pred = sent_model.predict(padded)[0][0]
            sentiment = "Positive" if pred > 0.5 else "Negative"
            st.metric("Sentiment", sentiment, f"{pred if pred>0.5 else 1-pred:.1%}")
            if pred > 0.8: st.balloons()
            elif pred < 0.2: st.snow()