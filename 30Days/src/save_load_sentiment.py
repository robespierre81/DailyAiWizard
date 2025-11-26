import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# -------------------------------------------------
# 1. Train a small sentiment model (first run only)
# -------------------------------------------------
def train_and_save():
    print("Training sentiment model for the first time...")
    vocab_size = 10000
    maxlen = 200
    
    # Load IMDB (num_words limits vocab)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)
    
    # Save in all 3 formats
    model.save('sentiment_model.h5')                    # ← Best for Streamlit
    model.save('sentiment_savedmodel', save_format='tf') # ← SavedModel format
    model.save_weights('sentiment_weights.h5')          # ← Weights only
    
    print("Model saved in 3 formats!")
    return model

# -------------------------------------------------
# 2. Load and predict (instant on every restart)
# -------------------------------------------------
if not os.path.exists('sentiment_model.h5'):
    model = train_and_save()
else:
    print("Loading saved model instantly...")
    model = tf.keras.models.load_model('sentiment_model.h5')  # ← ONE LINE!

# Test prediction
sample = np.zeros((1, 200))
sample[0, 50:55] = [1, 14, 20, 50, 100]  # fake positive words
pred = model.predict(sample)[0][0]
print(f"Sample prediction: {'Positive' if pred > 0.5 else 'Negative'} ({pred:.3f})")