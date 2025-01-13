from ultralytics import YOLO
from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%Y%m%d')
project_name = "PS_CLS"
project_sub_name = f"{formatted_date}_aug"

data_set_dir = "/data/proj/zhangx/B19RPA/PS_CLS/20241223_3class"

model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights

results = model.train(data=data_set_dir, 
                      epochs=100, 
                      imgsz=640, 
                      device=1,
                      project=f'runs/{project_name}',
                      name=project_sub_name)  

# # Load the model
# model = YOLO("/home/tvt/zhangx/study/runs/train/weights/best.pt")

# # Export the model to ONNX format
# model.export(format="onnx")
