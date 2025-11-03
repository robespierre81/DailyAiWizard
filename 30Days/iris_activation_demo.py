import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def iris_activation_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    activations = ['relu', 'sigmoid', 'tanh', 'softmax']
    results = {}
    
    for act in activations:
        model = Sequential([
            Dense(16, activation=act, input_shape=(4,)),
            Dense(16, activation=act),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results[act] = test_acc
    
    print("Activation Function Results:")
    for act, acc in results.items():
        print(f"{act.upper()}: {acc:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Activation Function Performance')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.show()

iris_activation_demo()