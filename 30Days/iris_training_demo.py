import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def iris_training_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = Sequential([
        Dense(16, activation='relu', input_shape=(4,)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_iris_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    best_model = tf.keras.models.load_model('best_iris_model.h5')
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Best Model Test Accuracy: {test_acc:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

iris_training_demo()