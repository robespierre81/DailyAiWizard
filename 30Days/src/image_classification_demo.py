import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def image_classification_demo():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2, verbose=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualize
    preds = model.predict(X_test[:10])
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(X_test[i])
        plt.title(class_names[np.argmax(preds[i])])
        plt.axis('off')
    plt.show()

image_classification_demo()