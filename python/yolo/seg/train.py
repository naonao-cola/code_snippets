from ultralytics import YOLO

# Load a model
data_path = r"/data/lb/ultralytics/ultralytics/cfg/datasets/my_data.yaml"
model_path = r"/data/lb/ultralytics/yolov8m-seg.pt"
cfg_path = f"/data/lb/ultralytics/ultralytics/cfg/default.yaml"

model = YOLO(model_path)

# Use the model
model.train(data=data_path, cfg=cfg_path)
