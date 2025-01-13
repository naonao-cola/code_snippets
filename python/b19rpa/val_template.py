from ultralytics import YOLO

# 加载模型

# model_path = "/home/zhangx/project/B19RPA/runs/BM/20250103_11s/exp_aug2/weights/best.pt"
# model = YOLO(model_path)  # 替换为你的模型路径

# 进行评估
# metrics = model.predict(source='/home/zhangx/dataset/B19RPA/BM/20241209_BM_dataset/val',
#                         imgsz=640,
#                         batch=32,
#                         device=0,
#                         # save_txt=True,
#                         # save_conf=True,
#                         # save_json=True,
#                         # project="./runs/predict"
#                         )


model_path = "/home/zhangx/project/B19RPA/runs/TVPS/20250106_11s/exp_aug/weights/best.pt"
model = YOLO(model_path)  # 替换为你的模型路径
# Validate the model
metrics = model.val(data='/home/zhangx/project/B19RPA/TVPS/dataset_TVPS.yaml', 
                    iou=0.6, 
                    conf=0.25, 
                    device=0, 
                    save_json=True)