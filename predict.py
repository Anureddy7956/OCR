import os
import sys
import cv2
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Identify background and invert if necessary
    # Assuming text is darker than background if mean > 127
    if np.mean(img) > 127:
        img = 255 - img
        
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    model_path = 'model/mnist_cnn.keras'
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        sys.exit(1)
        
    model = tf.keras.models.load_model(model_path)
    
    try:
        processed_img = preprocess_image(image_path)
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit]
        
        print(f"Predicted Digit: {digit}")
        print(f"Confidence Score: {confidence*100:.2f}%")
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == '__main__':
    main()
