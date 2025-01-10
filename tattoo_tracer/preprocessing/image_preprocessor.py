"""
Image Preprocessor for Tattoo Trace Project

This module is designed to preprocess images for the Tattoo Trace project. It enhances image quality,
detects people using YOLOv8, and crops detected people for further analysis.

Author  : Felipe Camargo de Pauli
Email   : fcdpauli@gmail.com
Advisor : Prof. Dr. Heitor Silvério Lopes
Date    : February 2024
Github  : https://github.com/felipedepauli/tattoo_trace
"""

import cv2
import argparse
from ultralytics import YOLO

class ImagePreprocessor:
    """
    A class to preprocess images by enhancing quality, detecting people, and cropping them from images.
    """
    def __init__(self, model_path):
        """
        Initializes the ImagePreprocessor with necessary configurations and models.
        """
        # Load YOLOv8 model for person detection
        self.model = YOLO(model_path)

    def xyxy_to_xywh(self, bbox):
        """
        Converts bounding box from xyxy to xywh format.
        
        Parameters:
        - bbox: A list containing the bounding box in xyxy format.
        
        Returns:
        - A list containing the bounding box in xywh format.
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1]

    def locate_people(self, img, class_to_locate=0):
        """
        Detects people in the image using the YOLOv8 model.
        
        Parameters:
        - img: The image in which to detect people.
        
        Returns:
        - people_bboxes: A list of bounding boxes for detected people.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, verbose=False)
        people_bboxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                if cls == class_to_locate:  # Class 0 for people
                    bbox_xywh = self.xyxy_to_xywh([x1, y1, x2, y2])
                    people_bboxes.append((bbox_xywh, float(conf)))
              
        num_of_people = len(people_bboxes)
        if num_of_people == 0:
            print("No people detected in the image.")
        elif num_of_people == 1:
            print("1 person detected in the image.")
        else:
            print(f"{len(people_bboxes)} people detected in the image.") 
        return people_bboxes

    def crop_people(self, img, bboxes, output_size=(224, 224)):
        """
        Crops detected people from the image based on bounding boxes.
        
        Parameters:
        - img: The image from which to crop people.
        - bboxes: A list of bounding boxes for detected people.
        
        Returns:
        - crops: A list of cropped images, each containing a detected person.
        """
        crops = []
        for bbox, conf in bboxes:
            if conf > 0.5:
                x_center, y_center, w, h = bbox
                x, y = x_center - w // 2, y_center - h // 2
                crop = img[max(0, y):y+h, max(0, x):x+w]
                if crop.size > 0:
                    crops.append(crop)
        return crops
    
    def resize_crops(self, crops_original):
        """
        Resizes the cropped images to a specified size.
        
        Parameters:
        - crops_original: A list of cropped images.
        
        Returns:
        - A list of resized cropped images.
        """
        return [cv2.resize(crop, (224, 224)) for crop in crops_original]

def main():
    """
    Main function to execute the image preprocessing with command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Preprocessing for Tattoo Trace Project.")
    parser.add_argument("--locate", action='store_true', help="Locate people using YOLOv8 and process images.")
    parser.add_argument("--resize", nargs=2, type=int, help="Resize images to the specified width and height, e.g., --resize 224 224", default=(224, 224))
    parser.add_argument("image_path", type=str, help="Path to the image to be processed.")
    args = parser.parse_args()

    model_path = 'yolov8n.pt'
    preprocessor = ImagePreprocessor(model_path)

    if args.locate:
        img = cv2.imread(args.image_path)
        people_bboxes = preprocessor.locate_people(img)
        crops = preprocessor.crop_people(img, people_bboxes, output_size=tuple(args.resize))
        for i, crop in enumerate(crops):
            cv2.imshow(f"Crop {i+1}", crop)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()