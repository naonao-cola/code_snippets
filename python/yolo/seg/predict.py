from ultralytics import YOLO

# Load a model
# model = YOLO('/data/lb/hf_dataset/test/train5/weights/best.pt')
model = YOLO('/data/lb/hf_dataset/v1/out15/weights/best.pt', task="segment")
# Run batched inference on a list of images
model.predict("/data/lb/hf_dataset/1119_template/gray_far/11.jpg", save=True, device=0)
