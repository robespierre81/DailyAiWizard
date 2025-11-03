import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

def iris_loss_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot for categorical_crossentropy
    encoder = OneHotEncoder(sparse=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    losses = ['sparse_categorical_crossentropy', 'categorical_crossentropy']
    results = {}
    
    for loss in losses:
        model = Sequential([
            Dense(16, activation='relu', input_shape=(4,)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        if loss == 'categorical_crossentropy':
            history = model.fit(X_train, y_train_onehot, epochs=50, validation_split=0.2, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
        else:
            history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        results[loss] = test_acc
    
    print("Loss Function Results:")
    for loss, acc in results.items():
        print(f"{loss}: {acc:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Loss Function Performance')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.show()

iris_loss_demo()