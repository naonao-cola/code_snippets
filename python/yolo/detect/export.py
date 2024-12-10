from ultralytics import YOLO
from pathlib import Path
from pathlib import Path
from PIL import Image
import glob
import json

model_path = "/data/proj/www/repo/yolo8_test/src/yolov8s.pt"


model = YOLO(model_path)
model.export(format="onnx", dynamic=True)
