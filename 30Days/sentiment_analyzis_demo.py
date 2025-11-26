import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

def sentiment_analysis_demo():
    vocab_size = 10000
    maxlen = 200
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    
    model = Sequential([
        Embedding(vocab_size, 128, input_length=maxlen),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualize
    sample_review = X_test[0:1]
    pred = model.predict(sample_review)[0][0]
    print(f"Sample Prediction: {'Positive' if pred > 0.5 else 'Negative'} ({pred:.2f})")

sentiment_analysis_demo()