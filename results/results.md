# Benchmark results

- Date: 2026-11-08
- CPU: Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz (8 threads)
- OS: Linux-7.0.0-42-generic-x86_64-with-glibc2.39
- Python: 3.12.3, OpenCV: 4.14.0
- Clip: `data/short.mp4`, 242 frames measured after a 10-frame warm-up

| Detector | Mean ms/frame | 95th percentile ms | FPS | Avg boxes/frame |
| --- | --- | --- | --- | --- |
| YOLO11n PyTorch, imgsz 640 | 66.6 | 93.7 | 15.0 | 1.00 |
| YOLO11n PyTorch, imgsz 320 | 29.1 | 39.9 | 34.4 | 1.00 |
| YOLO11n ONNX Runtime, imgsz 320 | 25.3 | 27.3 | 39.5 | 1.00 |
| Haar face detector (baseline) | 306.1 | 337.8 | 3.3 | 1.40 |

Notes: the YOLO rows detect person, laptop and cell phone; the Haar row detects frontal faces only,
so compare speed across rows, not the box counts. Times include only `detect()`, not drawing or display.
