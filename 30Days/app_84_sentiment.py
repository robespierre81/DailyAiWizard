# day82_train_sentiment_model.py
# Run once → creates sentiment_model.h5 (IMDB reviews)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Loading IMDB dataset...")
vocab_size = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential([
    Embedding(vocab_size, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Training sentiment model...")
model.fit(x_train, y_train,
          epochs=10,
          batch_size=128,
          validation_split=0.2,
          verbose=1)

# Save the final model
model.save("sentiment_model.h5")
print("sentiment_model.h5 saved successfully! (~87–89% accuracy)")