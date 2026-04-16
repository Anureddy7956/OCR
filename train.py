import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1)),
        ReLU(),
        MaxPool2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3)),
        ReLU(),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128),
        ReLU(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Loading dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    model = create_model()
    model.summary()
    
    print("Training the model...")
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
    
    model.save('model/mnist_cnn.keras')
    print("Model saved to model/mnist_cnn.keras")
    
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model/confusion_matrix.png')
    print("Confusion matrix saved to model/confusion_matrix.png")

if __name__ == '__main__':
    main()
