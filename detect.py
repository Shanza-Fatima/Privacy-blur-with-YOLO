import cv2
from detectors import YoloDetector, HaarFaceDetector

cap = cv2.VideoCapture(0)                     # open the webcam
ok, frame = cap.read()                        # take one picture
cap.release()                                 # close the webcam

print("YOLO:", YoloDetector().detect(frame))  # e.g. [(120, 80, 520, 470)] if you are in view
print("Haar:", HaarFaceDetector().detect(frame))  # e.g. [(250, 120, 380, 250)] if your face is visible