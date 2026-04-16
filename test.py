import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    model_path = 'model/mnist_cnn.keras'
    if not os.path.exists(model_path):
        print("Model not found. Please run train.py first.")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('model/test_confusion_matrix.png')
    print("Test confusion matrix saved to model/test_confusion_matrix.png")

if __name__ == '__main__':
    main()
