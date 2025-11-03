import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def iris_optimizer_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    optimizers = {
        'sgd': tf.keras.optimizers.SGD(),
        'momentum': tf.keras.optimizers.SGD(momentum=0.9),
        'adam': tf.keras.optimizers.Adam(),
        'rmsprop': tf.keras.optimizers.RMSprop()
    }
    results = {}
    
    for name, opt in optimizers.items():
        model = Sequential([
            Dense(16, activation='relu', input_shape=(4,)),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results[name] = test_acc
    
    print("Optimizer Results:")
    for name, acc in results.items():
        print(f"{name.upper()}: {acc:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Optimizer Performance')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.show()

iris_optimizer_demo()