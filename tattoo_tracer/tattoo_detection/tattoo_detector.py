"""
Tattoo Detector for Tattoo Trace Project

This module is designed to detect tattoos in images using machine learning models.
It provides functionalities to preprocess images, including resizing to a maximum dimension to ensure
consistency and improve processing speed, detect tattoos, and optionally display the detection results with bounding boxes.

Author  : Felipe Camargo de Pauli
Email   : fcdpauli@gmail.com
Advisor : Prof. Dr. Heitor Silvério Lopes
Date    : February 2024
Github  : https://github.com/felipedepauli/tattoo_trace
"""

# Import necessary libraries and modules
import cv2
import argparse
from tensorflow.keras.models import load_model
import numpy as np

class TattooDetector:
    """
    A class to detect tattoos in images using a trained machine learning model.
    """
    def __init__(self, model_path='detector_model_000.h5'):
        """
        Initializes the TattooDetector with a trained model.
        """
        self.model = load_model(model_path)

    def detect_tattoo(self, img):
        """
        Detects tattoos in the given image using the loaded model.

        Parameters:
        - img: The preprocessed image in which to detect tattoos.

        Returns:
        - A list of bounding boxes (x, y, w, h) for each detected tattoo and their probabilities.
        """
        # Validate the input image
        if img is None or not hasattr(img, 'shape'):
            print("Invalid image passed to detect_tattoo.")
            return [], 0  # Return an empty list and 0 probability for invalid input

        # Ensure that the image is in the correct format (224x224x3) and normalized
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Adds a dimension for the batch

        # Make the prediction
        predictions = self.model.predict(img_array)
        print("Predictions:", predictions)
        
        confidence = predictions[0][0]
        
        return [(50, 300, 200, 100)], confidence  # Example of bounding box

    def preprocess_image(self, image_path, max_size=800):
        """
        Loads and preprocesses the image from the given path, including resizing if the image exceeds the maximum size.

        Parameters:
        - image_path: The path to the image file.
        - max_size: The maximum size for the longest dimension of the image.

        Returns:
        - The preprocessed image.
        """
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        # Resize image if either dimension is larger than max_size
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            
        return img

    def display_detection(self, img, bboxes):
        """
        Displays the image with detected tattoos highlighted by bounding boxes.

        Parameters:
        - img: The image with detected tattoos.
        - bboxes: A list of bounding boxes for each detected tattoo.
        """
        if isinstance(bboxes, tuple):
            bboxes = [bboxes]
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Detected Tattoos", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """
    Main function to execute the tattoo detection with command line arguments.
    """
    parser = argparse.ArgumentParser(description="Detect tattoos in an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    detector = TattooDetector()
    img = detector.preprocess_image(args.image_path)
    bboxes, prob = detector.detect_tattoo(img)
    if prob > 0:
        print(f'There is a tattoo with probability {prob:.2f} in the image.')
        detector.display_detection(img, bboxes)
    else:
        print('No tattoos detected in the image.')

if __name__ == "__main__":
    main()
