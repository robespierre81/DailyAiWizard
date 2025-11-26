import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("image_model.h5")

model = load_model()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("Day 84 – Hugging Face Spaces Live Demo")
st.write("Upload any image — AI classifies it instantly using Day 80 model!")

uploaded = st.file_uploader("Choose an image...", type=['png','jpg','jpeg'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    img_resized = img.resize((32,32))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1,32,32,3)
    
    pred = model.predict(img_array)[0]
    predicted_class = np.argmax(pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_column_width=True)
    with col2:
        st.success(f"**{class_names[predicted_class].capitalize()}**")
        st.bar_chart(pred, height=300)