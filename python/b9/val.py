from ultralytics import YOLO

# Load a model
model = YOLO("runs/test2/weights/best.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model("/home/zhangx/dataset/B9/20250106_B_dataset/val",
                save = True,
                save_txt = True,
                save_conf = True)  # return a list of Results objects


# from sahi import AutoDetectionModel
# from sahi.predict import predict

# predict(
#     model_type="ultralytics",
#     model_path="runs/test/weights/best.pt",
#     model_device="cuda:0",  # or 'cuda:0'
#     model_confidence_threshold=0.8,
#     source="/home/zhangx/dataset/B9/20250106_B_dataset/val",
#     slice_height=640,
#     slice_width=640,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )