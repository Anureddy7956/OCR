import os
import sys
import cv2
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def process_image(image_path, model_path='model/mnist_cnn.keras', save_path=None):
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return None
        
    model = tf.keras.models.load_model(model_path)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img.copy()
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 5 and h >= 15: # Filter out very small noise
            roi = gray[y:y+h, x:x+w]
            
            roi_resized = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
            
            if np.mean(roi_resized) > 127:
                roi_resized = 255 - roi_resized
                
            padded = np.pad(roi_resized, ((4,4),(4,4)), 'constant', constant_values=0)
            
            processed = padded.reshape(1, 28, 28, 1).astype('float32') / 255.0
            
            pred = model.predict(processed, verbose=0)
            digit = np.argmax(pred[0])
            
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(output_img, str(digit), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
    if save_path:
        cv2.imwrite(save_path, output_img)
        print(f"Saved OCR result to {save_path}")
        
    return output_img

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    save_path = "ocr_output.png"
    process_image(image_path, save_path=save_path)

if __name__ == '__main__':
    main()
