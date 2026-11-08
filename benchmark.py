#measure speed of every detector on the same clip
# benchmark.py
# Times every detector on the SAME video clip and writes a results table.
# Run:  python benchmark.py --video data/sample.mp4

import argparse                              # options typed after the command
import datetime                              # today's date for the results file
import os                                    # check files exist, make folders
import platform                              # computer / Python information
import time                                  # the precise stopwatch

import cv2                                   # reads the video
import numpy as np                           # mean and percentile calculations

from detectors import YoloDetector, HaarFaceDetector   # the detectors we compare


def time_detector(detector, video_path, max_frames=300, warmup=10):
    """Run `detector` on the clip and return timing statistics."""
    cap = cv2.VideoCapture(video_path)                       # open the clip
    if not cap.isOpened():
        raise SystemExit(f"Could not open the clip: {video_path}")

    times, counts = [], []                                   # seconds per frame, and boxes per frame
    index = 0                                                # which frame we are on
    while index < warmup + max_frames:                       # warm-up frames first, then the real ones
        ok, frame = cap.read()                               # next picture
        if not ok:                                           # clip ended
            break
        start = time.perf_counter()                          # stopwatch on
        boxes = detector.detect(frame)                       # the thing we are measuring
        elapsed = time.perf_counter() - start                # stopwatch off
        if index >= warmup:                                  # the first frames are slow (loading), so skip them
            times.append(elapsed)
            counts.append(len(boxes))
        index += 1
    cap.release()

    if not times:
        raise SystemExit("The clip is too short: record at least a few seconds.")

    ms = np.array(times) * 1000.0                            # seconds -> milliseconds
    return {
        "mean_ms": float(ms.mean()),                         # average time per frame
        "p95_ms": float(np.percentile(ms, 95)),              # 95% of frames were faster than this (shows slow spikes)
        "fps": float(1000.0 / ms.mean()),                    # frames per second = 1000 ms / ms per frame
        "avg_boxes": float(np.mean(counts)),                 # average detections per frame
        "frames": len(ms),                                   # how many frames were measured
    }


def cpu_name():
    """Read the CPU model on Linux, so the README can say what hardware the numbers came from."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    p = argparse.ArgumentParser(description="Benchmark the detectors on one clip")
    p.add_argument("--video", default="data/short.mp4", help="the clip to test on")
    p.add_argument("--frames", type=int, default=300, help="how many frames to measure")
    p.add_argument("--out", default="results/results.md", help="where to write the table")
    args = p.parse_args()

    # Each entry: (label shown in the table, a function that builds the detector).
    # "lambda: ..." delays loading, so one model is loaded at a time.
    configs = [
        ("YOLO11n PyTorch, imgsz 640", lambda: YoloDetector("yolo11n.pt", imgsz=640)),
        ("YOLO11n PyTorch, imgsz 320", lambda: YoloDetector("yolo11n.pt", imgsz=320)),
    ]
    if os.path.exists("yolo11n.onnx"):                       # only if you ran export_onnx.py
        configs.append(("YOLO11n ONNX Runtime, imgsz 320", lambda: YoloDetector("yolo11n.onnx", imgsz=320)))
    else:
        print("yolo11n.onnx not found: skipping ONNX (run export_onnx.py first)")
    configs.append(("Haar face detector (baseline)", lambda: HaarFaceDetector()))

    rows = []
    for label, build in configs:                             # test each detector in turn
        print(f"Running: {label} ...")
        detector = build()                                   # load it
        stats = time_detector(detector, args.video, max_frames=args.frames)
        rows.append((label, stats))                          # remember the numbers
        print(f"  {stats['fps']:.1f} FPS, {stats['mean_ms']:.1f} ms/frame")

    lines = [                                                # build the markdown table, line by line
        "# Benchmark results",
        "",
        f"- Date: {datetime.date.today()}",
        f"- CPU: {cpu_name()} ({os.cpu_count()} threads)",
        f"- OS: {platform.platform()}",
        f"- Python: {platform.python_version()}, OpenCV: {cv2.__version__}",
        f"- Clip: `{args.video}`, {rows[0][1]['frames']} frames measured after a 10-frame warm-up",
        "",
        "| Detector | Mean ms/frame | 95th percentile ms | FPS | Avg boxes/frame |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, s in rows:
        lines.append(f"| {label} | {s['mean_ms']:.1f} | {s['p95_ms']:.1f} | {s['fps']:.1f} | {s['avg_boxes']:.2f} |")
    lines += [
        "",
        "Notes: the YOLO rows detect person, laptop and cell phone; the Haar row detects frontal faces only,",
        "so compare speed across rows, not the box counts. Times include only `detect()`, not drawing or display.",
        "",
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)   # make results/ if missing
    with open(args.out, "w") as f:                               # write the file
        f.write("\n".join(lines))
    print("\n".join(lines))                                      # and show it on screen


if __name__ == "__main__":
    main()