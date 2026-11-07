#Yolo dectector and HaarFaceDectector, same interface
# Two detectors with the SAME method: detect(frame) returns a list of boxes (x1, y1, x2, y2).
# Because they look the same from outside, other files can use either one.

import cv2                                   # needed for the Haar face detector
from ultralytics import YOLO                 # the YOLO model loader


class YoloDetector:
    """Finds people, laptops and phones (or any classes you ask for) with a YOLO model."""

    def __init__(self, weights="yolo11n.pt", imgsz=320, conf=0.4,
                 wanted=("person", "laptop", "cell phone")):
        self.model = YOLO(weights, task="detect")  # load the model; downloads yolo11n.pt the first time
        self.imgsz = imgsz                         # picture size the model sees: smaller = faster, less exact
        self.conf = conf                           # ignore detections the model is less than 40% sure about

        # model.names maps class numbers to names, e.g. {0: 'person', 63: 'laptop', 67: 'cell phone', ...}
        # We turn the NAMES we want into class NUMBERS, so no magic numbers appear in the code.
        self.target_ids = {number for number, name in self.model.names.items() if name in wanted}
        if not self.target_ids:                    # none of the names exist in this model
            raise ValueError(f"None of {wanted} are class names of this model")

    def detect(self, frame):
        # Run the model on one picture. [0] picks the result for our single picture.
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    device="cpu", verbose=False)[0]

        boxes = []                                 # the answer we will return
        corners = result.boxes.xyxy.cpu().numpy()  # N rows of (x1, y1, x2, y2), converted to a NumPy table
        classes = result.boxes.cls.cpu().numpy()   # N class numbers, in the same order
        for (x1, y1, x2, y2), cls in zip(corners, classes):   # walk through the detections one by one
            if int(cls) in self.target_ids:        # keep only the classes we want to blur
                boxes.append((int(x1), int(y1), int(x2), int(y2)))  # whole numbers, because pixels are whole
        return boxes


class HaarFaceDetector:
    """The old-style face detector that ships inside OpenCV. We use it as a baseline."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(40, 40)):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # file of face patterns inside OpenCV
        self.cascade = cv2.CascadeClassifier(path)   # load those patterns
        if self.cascade.empty():                     # loading failed
            raise RuntimeError(f"Could not load Haar file: {path}. "
                               "Use OpenCV 4 (pip install 'opencv-python<5'); version 5 does not ship it.")
        self.scale_factor = scale_factor             # how much the search window grows each round (1.1 = 10%)
        self.min_neighbors = min_neighbors           # how many overlapping hits make a real face (higher = fewer false alarms)
        self.min_size = min_size                     # ignore faces smaller than 40 x 40 pixels

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Haar works on brightness only, so drop the colour
        faces = self.cascade.detectMultiScale(           # search the picture at many sizes
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=self.min_size)
        # Haar returns (x, y, width, height). We convert to corners (x1, y1, x2, y2) to match YoloDetector.
        return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]