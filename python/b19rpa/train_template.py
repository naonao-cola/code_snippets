# tensorboard --logdir runs

from ultralytics import YOLO
from datetime import datetime

project_dir = "/home/tvt/zhangx/project/B19RPA"
  
def my_yolo_det_train(project_name, yolo_model="v8s", epochs = 200, batch = 64, device = 1, is_aug=True):
    # 获取当前日期
    current_date = datetime.now()
    # 格式化日期为 "YYYYMMDD"
    formatted_date = current_date.strftime('%Y%m%d')
    # yolo_model = "v8s"
    # project_name = "ITO"
    model_name = f"yolo{yolo_model}.pt"
    data_set = f"{project_dir}/{project_name}/dataset_{project_name}.yaml"

    project_sub_name = f"{formatted_date}_{yolo_model}/exp_aug" if is_aug else f"{formatted_date}_{yolo_model}/exp"

    # Load a model
    model = YOLO(f"{project_dir}/models/{model_name}")  # build a new model from scratch

    # train
    model.train(data = data_set,
                cache = False,
                imgsz = 640,
                epochs = epochs,
                batch = batch,
                close_mosaic = 50,
                workers = 8,
                device = device,
                # optimizer='SGD', # using SGD
                # amp=False, # close amp
                project=f'{project_dir}/runs/{project_name}',
                name=project_sub_name,
                cfg = f"{project_dir}/{project_name}/config.yaml",
                patience=1000,
                )
