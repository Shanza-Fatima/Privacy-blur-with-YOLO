#record a 20-second test clip from your webcamENTICAL frames.
# Run:  python record_clip.py --seconds 20      (stand, sit, hold up your phone, walk around)

import argparse                              # reads options typed after the command
import os                                    # to create the output folder
import time                                  # to stop after N seconds

import cv2                                   # camera and video file writing


def main():
    p = argparse.ArgumentParser(description="Record a test clip from the webcam")
    p.add_argument("--camera", default="0", help="camera number or a video file path")
    p.add_argument("--seconds", type=int, default=20, help="how long to record")
    p.add_argument("--out", default="data/sample.mp4", help="where to save the clip")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = p.parse_args()

    source = int(args.camera) if args.camera.isdigit() else args.camera  # camera number or file path
    cap = cv2.VideoCapture(source)                                       # open it
    if not cap.isOpened():
        raise SystemExit(f"Could not open: {args.camera}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30                                # speed the camera reports (30 if it reports 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                       # picture width in pixels
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                     # picture height in pixels

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)         # make the folder if it is missing
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),  # mp4v = a video format every player opens
                             fps, (width, height))

    start = time.time()                                                  # when we began
    frames = 0                                                           # how many pictures saved
    while time.time() - start < args.seconds:                            # keep going until time is up
        ok, frame = cap.read()                                           # grab a picture
        if not ok:                                                       # nothing came: stop
            break
        writer.write(frame)                                              # save it into the clip
        frames += 1
        if not args.no_show:
            left = int(args.seconds - (time.time() - start))             # whole seconds remaining
            preview = frame.copy()                                       # draw on a copy so the saved clip stays clean
            cv2.putText(preview, f"REC {left}s (q = stop)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)   # red text
            cv2.imshow("recording", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):                        # q stops early
                break

    cap.release()                                                        # give the camera back
    writer.release()                                                     # finish the file
    cv2.destroyAllWindows()
    print(f"Saved {frames} frames to {args.out}")


if __name__ == "__main__":
    main()