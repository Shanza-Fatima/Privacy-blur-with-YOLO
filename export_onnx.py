#converts the Yolo model to onnx
# export_onnx.py
# Saves the YOLO model in ONNX format, so ONNX Runtime can run it.
# Run once:  python export_onnx.py     -> creates yolo11n.onnx next to yolo11n.pt

from ultralytics import YOLO                          # the model loader

model = YOLO("yolo11n.pt")                            # load the normal PyTorch model
path = model.export(format="onnx", imgsz=320)         # convert it; imgsz=320 is fixed inside the file
print("Saved:", path)                                 # shows where the .onnx file was written