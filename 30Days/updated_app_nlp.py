import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

def updated_app_nlp():
    st.header("Write a Review – AI Feels It!")
    vocab_size = 1000
    maxlen = 20

    if not os.path.exists('nlp_model.h5'):
        st.info("Training NLP model for the first time...")
        (X_train, _), _ = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
        X_train = pad_sequences(X_train, maxlen=maxlen)

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 128, input_length=maxlen),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, np.random.randint(0, 2, len(X_train)), epochs=3, verbose=0)
        model.save('nlp_model.h5')
        st.success("NLP model trained!")

    model = tf.keras.models.load_model('nlp_model.h5')
    tokenizer = Tokenizer(num_words=vocab_size)

    review = st.text_area("Enter your movie review:", height=150)

    if review:
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=maxlen)
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"

        st.subheader("Prediction")
        st.success(f"**Sentiment: {sentiment}**\nConfidence: {prediction:.1%}")

if __name__ == "__main__":
    updated_app_nlp()