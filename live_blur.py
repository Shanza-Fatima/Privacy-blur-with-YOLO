#the main program: webcam -> detect -> blur -> show
import argparse                              # reads options typed after the command
import time                                  # a precise clock, to measure FPS

import cv2                                   # camera, drawing text, windows, video file writing

from blur_utils import blur_box              # our blur function (file 1)
from detectors import YoloDetector           # our YOLO detector (file 2)


def parse_args():
    p = argparse.ArgumentParser(description="Real-time privacy blur with YOLO")
    p.add_argument("--source", default="0", help="camera number (0) or path of a video file")
    p.add_argument("--weights", default="yolo11n.pt", help="model file: yolo11n.pt or yolo11n.onnx")
    p.add_argument("--imgsz", type=int, default=640, help="picture size the model sees (320 fast, 640 exact)")
    p.add_argument("--conf", type=float, default=0.25, help="minimum confidence, 0 to 1")
    p.add_argument("--strength", type=float, default=0.35, help="blur strength as a fraction of the box size (bigger = blurrier)")
    p.add_argument("--save", default=None, help="also write the result to this .mp4 file")
    p.add_argument("--save-fps", type=float, default=15, help="speed of the saved video")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    return p.parse_args()

def iou(a, b):                                                     # how much two boxes overlap: 0 = none, 1 = identical
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])                      # top-left corner of the shared area
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])                      # bottom-right corner of the shared area
    inter = max(0, x2 - x1) * max(0, y2 - y1)                      # area of the shared part (0 if they do not touch)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])                 # area of one box
    union = area(a) + area(b) - inter                              # area covered by either box
    return inter / union if union > 0 else 0.0


def update(tracks, found, max_age=5):                              # tracks: list of (box, frames since last seen)
    new = [(b, 0) for b in found]                                  # every fresh detection starts at age 0
    for box, age in tracks:                                        # look at each remembered box
        seen_again = any(iou(box, b) > 0.3 for b in found)         # is there a new box in about the same place?
        if not seen_again and age < max_age:                       # missed this frame, but not for too long
            new.append((box, age + 1))                             # keep it one more frame
    return new

def main():
    args = parse_args()                                            # read the options

    source = int(args.source) if args.source.isdigit() else args.source  # "0" -> camera 0; anything else -> file path
    cap = cv2.VideoCapture(source)                                 # open the camera or the video file
    if not cap.isOpened():                                         # camera busy, missing, or wrong file name
        raise SystemExit(f"Could not open the source: {args.source}")

    detector = YoloDetector(args.weights, imgsz=args.imgsz, conf=args.conf)  # load the model once, before the loop

    writer = None                                                  # created on the first frame, if --save is used
    fps_smooth = 0.0                                               # FPS number shown on screen
    tracks = []                                                    # boxes to blur now, and frames since the last detection
    if not args.no_show:                                           # only needed when a window is shown
        cv2.namedWindow("privacy blur (press q to quit)", cv2.WINDOW_NORMAL)  # allow resizing the window
        cv2.resizeWindow("privacy blur (press q to quit)", 1280, 720)         # start at a bigger size
    while True:                                                    # one round of this loop = one picture
        ok, frame = cap.read()                                     # grab a picture; ok is False if none came
        if not ok:                                                 # video ended, or the camera stopped
            break

        start = time.perf_counter()
        tracks = update(tracks, detector.detect(frame))
        boxes = [b for b, _ in tracks]
        for x1, y1, x2, y2 in boxes:
            side = max(x2 - x1, y2 - y1)
            k = min(max(int(side * args.strength), 15), 201)
            blur_box(frame, x1, y1, x2, y2, k)
        elapsed = time.perf_counter() - start                    # stop the stopwatch (seconds)                     # stop the stopwatch (seconds)

        fps_now = 1.0 / elapsed if elapsed > 0 else 0.0            # frames per second = 1 / seconds per frame
        fps_smooth = fps_now if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps_now  # smooth it so the number does not jitter

        label = f"{fps_smooth:.1f} FPS | {len(boxes)} blurred"     # text to show, e.g. "17.3 FPS | 2 blurred"
        cv2.putText(frame, label, (10, 30),                        # draw it near the top-left corner
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # font, size, colour (BGR green), thickness

        if args.save:                                              # the user asked to save a video
            if writer is None:                                     # first frame: now we know the picture size
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.save_fps, (width, height))
            writer.write(frame)                                    # add this picture to the video file

        if not args.no_show:                                       # show a window unless told not to
            cv2.imshow("privacy blur (press q to quit)", frame)    # display the picture
            if cv2.waitKey(1) & 0xFF == ord("q"):                  # wait 1 ms for a key; q means quit
                break

    cap.release()                                                  # give the camera back
    if writer is not None:
        writer.release()                                           # finish writing the video file
        print(f"Saved video to {args.save}")
    cv2.destroyAllWindows()                                        # close the window


if __name__ == "__main__":                                         # only run main() when started directly
    main()