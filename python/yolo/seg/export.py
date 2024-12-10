from ultralytics import YOLO


model = YOLO("/data/lb/hf_dataset/v1/out16/weights/best.pt")
model.export(format='onnx')
