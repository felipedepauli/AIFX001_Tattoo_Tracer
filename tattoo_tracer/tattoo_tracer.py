"""
Tattoo Tracer

This script integrates image preprocessing, pose detection, and tattoo detection to identify tattoos on specific body parts. It utilizes OpenCV for image processing, MediaPipe for pose detection, and a custom tattoo detection algorithm.

Usage:
    python tattoo_tracer.py <path_to_image>

Example:
    python tattoo_tracer.py images/example.jpg

Author  : Felipe Camargo de Pauli
Email   : fcdpauli@gmail.com
Advisor : Prof. Dr. Heitor Silvério Lopes
Date    : February 2024
Github  : https://github.com/felipedepauli/tattoo_trace
"""

# Import necessary libraries and modules
import cv2
import argparse
from preprocessing.image_preprocessor import ImagePreprocessor
from pose_detection.pose_detector import PoseDetector
from tattoo_detection.tattoo_detector import TattooDetector
from pose_detection.body_parts import BodyPart, simplify_body_parts, simplified_body_parts_mapping

class TattooTracer:
    """
    A class to trace tattoos in images by integrating pose detection with tattoo detection.
    """
    def __init__(self):
        """
        Initializes the TattooTracer with necessary components for image preprocessing, pose detection, and tattoo detection.
        """
        self.image_preprocessor = ImagePreprocessor('yolov8n.pt')
        self.tattoo_detector = TattooDetector('model_path.h5')
        self.pose_detector = PoseDetector()

    def is_point_inside_bbox(self, point, bbox):
        """
        Checks if a given point is inside a bounding box.

        Parameters:
        - point (tuple): The (x, y) coordinates of the point.
        - bbox (tuple): The bounding box (x, y, w, h).

        Returns:
        - bool: True if the point is inside the bbox, False otherwise.
        """
        px, py = point
        bx, by, bw, bh = bbox
        return bx <= px <= bx + bw and by <= py <= by + bh

    def trace_tattoos(self, image_path):
        """
        Traces tattoos in the given image by integrating image preprocessing, pose detection, and tattoo detection.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - None
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found at {image_path}")
            return
        
        # Resize the image if the height is greater than 800
        height, width = img.shape[:2]
        if height > 800:
            # Calculate the new width to maintain the aspect ratio
            new_width = int((800 / height) * width)
            img = cv2.resize(img, (new_width, 800))
        
        # Step 1: Locate and crop all people from the original image
        people_bboxes = self.image_preprocessor.locate_people(img)
        crops_original = self.image_preprocessor.crop_people(img, people_bboxes)
        
        # Step 2: Create copies of the crops and resize for analysis
        crops_resized = self.image_preprocessor.resize_crops(crops_original)

        # Step 3: Pass resized crops through the model for tattoo detection
        threshold = 0.72
        tattoo_predictions = [self.tattoo_detector.detect_tattoo(crop) for crop in crops_resized]

        # Step 5 and 6: Iterate through the original crops and check the detection result
        for i, (crop_original, prediction) in enumerate(zip(crops_original, tattoo_predictions)):
            if prediction > threshold:
                print(f"Crop {i+1}: Tattoo detected")
                tattoo_bbox = prediction  # Assuming this gives the bbox as (x, y, w, h)
                
                # Draw the bbox of the tattoo
                # Step 7: If tattoo detected, use PoseDetector to indicate where it is
                crop_with_pose = self.pose_detector.findPose(crop_original, draw=True)
                pose_landmarks = self.pose_detector.findPosition(crop_with_pose, draw=True)
                x, y, w, h = tattoo_bbox[0][0]
                cv2.rectangle(crop_with_pose, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(crop_with_pose, "Tattoo", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                

                # Detect pose in the crop_original image
                self.pose_detector.findPose(crop_original, draw=False)
                pose_landmarks = self.pose_detector.findPosition(crop_original, draw=False)
                
                # Check if any pose point is inside the tattoo bbox
                body_parts_inside_tattoo = []
                for id, cx, cy in pose_landmarks:
                    if x <= cx <= x+w and y <= cy <= y+h:  # Check if the point is inside the bbox
                        body_part = BodyPart(id)
                        if body_part in simplified_body_parts_mapping:
                            body_parts_inside_tattoo.append(simplified_body_parts_mapping[body_part])
                
                # Inform about affected body parts
                if body_parts_inside_tattoo:
                    print("\nBody parts with tattoo:")
                    for body_part in sorted(set(body_parts_inside_tattoo)):
                        print(f"- {body_part}")
                else:
                    print("No pose landmarks are inside the tattoo bbox.")          
            else:
                print(f"Crop {i+1}: No tattoo detected")

            # Show the image with the tattoo bbox drawn
            cv2.imshow(f"Crop {i+1} with Tattoo", crop_with_pose)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def line_intersects_rect(self, p1, p2, rect):
        """
        Check if a line segment intersects a rectangle.

        Args:
        - p1, p2: The endpoints of the line segment, each a tuple of (x, y).
        - rect: The rectangle, a tuple of (x, y, width, height).

        Returns:
        - bool: True if the line segment intersects the rectangle, False otherwise.
        """
        # This is a simplified approach. For a comprehensive solution, you may need to implement
        # an algorithm that checks for intersections between line segments and the rectangle edges.
        # Placeholder for simplicity. Implement based on your requirements.
        return True

def main():
    """
    Main function to execute the tattoo tracing process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trace tattoos in an image.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    # Create a TattooTracer object and trace the tattoos in the image
    tracer = TattooTracer()
    tracer.trace_tattoos(args.image_path)

if __name__ == "__main__":
    main()