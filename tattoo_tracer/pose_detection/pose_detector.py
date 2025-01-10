"""
Pose Detection Module

This module provides functionalities to detect human poses using MediaPipe and OpenCV.

Author  : Felipe Camargo de Pauli
Email   : fcdpauli@gmail.com
Advisor : Prof. Dr. Heitor Silvério Lopes
Date    : February 2024
Github  : 
"""

import cv2
import mediapipe as mp
from pose_detection.body_parts import BodyPart
# from body_parts import BodyPart
import argparse

class PoseDetector:
    """
    A class to detect human poses in images or video streams using MediaPipe.

    Attributes:
        mode (bool): Whether to treat the input images as a batch of static and possibly unrelated images, or a stream of images where detections from the previous images can be reused. Defaults to False.
        upBody (bool): Whether to focus the detection on the upper body of figures in the images. Defaults to False.
        smooth (bool): Whether to apply smoothing for the landmark positions across frames. Defaults to True.
        detectionCon (float): Minimum confidence value ([0.0, 1.0]) for the pose detection to be considered successful. Defaults to 0.5.
        trackCon (float): Minimum confidence value ([0.0, 1.0]) for the landmark tracking to be considered successful. Defaults to 0.5.
        mpDraw: MediaPipe drawing utility for drawing landmarks on images.
        mpPose: MediaPipe Pose solution.
        pose: An instance of the MediaPipe Pose solution configured with the specified attributes.
    """

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the PoseDetector object with custom configuration for pose detection.

        Parameters:
            mode (bool): See class attributes.
            upBody (bool): See class attributes.
            smooth (bool): See class attributes.
            detectionCon (float): See class attributes.
            trackCon (float): See class attributes.
        """
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        """
        Detects the human pose in an image and optionally draws the detected landmarks and connections.

        Parameters:
            img: The image in which to detect the pose.
            draw (bool): Whether to draw the detected landmarks and connections on the image. Defaults to True.

        Returns:
            The image with the detected pose landmarks and connections drawn if draw is True.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds the positions of the pose landmarks in the image.

        Parameters:
            img: The image from which to find the pose landmarks.
            draw (bool): Whether to draw the landmarks on the image. Defaults to True.

        Returns:
            A list of tuples, each containing the ID and the (x, y) coordinates of a landmark.
        """
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


    def getBodyPartCoordinates(self, img, bodyPart: BodyPart, draw=True):
        """
        Returns the coordinates of a specific body part.

        Parameters:
            img: The image from which to find the body part.
            bodyPart (BodyPart): The body part to find.
            draw (bool): Whether to draw the body part on the image. Defaults to True.

        Returns:
            A tuple (x, y) of the coordinates of the body part, or None if not found.
        """
        for id, cx, cy in self.findPosition(img, draw=False):
            if id == bodyPart.value:
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                return (cx, cy)
        return None
    
body_part_keys = {
    '0': BodyPart.NOSE,
    '1': BodyPart.LEFT_EYE,
    '2': BodyPart.RIGHT_EYE,
    '3': BodyPart.LEFT_EAR,
    '4': BodyPart.RIGHT_EAR,
    '5': BodyPart.LEFT_SHOULDER,
    '6': BodyPart.RIGHT_SHOULDER,
    '7': BodyPart.LEFT_ELBOW,
    '8': BodyPart.RIGHT_ELBOW,
    '9': BodyPart.LEFT_WRIST,
    'w': BodyPart.RIGHT_WRIST,
    'e': BodyPart.LEFT_PINKY,
    'r': BodyPart.RIGHT_PINKY,
    't': BodyPart.LEFT_INDEX,
    'y': BodyPart.RIGHT_INDEX,
    'u': BodyPart.LEFT_THUMB,
    'i': BodyPart.RIGHT_THUMB,
    'o': BodyPart.LEFT_HIP,
    'p': BodyPart.RIGHT_HIP,
    'a': BodyPart.LEFT_KNEE,
    's': BodyPart.RIGHT_KNEE,
    'd': BodyPart.LEFT_ANKLE,
    'f': BodyPart.RIGHT_ANKLE,
    'g': BodyPart.LEFT_HEEL,
    'h': BodyPart.RIGHT_HEEL,
    'j': BodyPart.LEFT_FOOT_INDEX,
    'k': BodyPart.RIGHT_FOOT_INDEX,
}



def main():
    
    parser = argparse.ArgumentParser(description="Real-time human pose detection with MediaPipe.")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    selected_body_part = None

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if selected_body_part is not None:
            # Obter as coordenadas da parte do corpo selecionada
            body_part_coords = detector.getBodyPartCoordinates(img, selected_body_part, draw=False)
            if body_part_coords:
                cv2.circle(img, body_part_coords, 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, selected_body_part.name, (body_part_coords[0] + 10, body_part_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Pose", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in body_part_keys:
            selected_body_part = body_part_keys[chr(key)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
