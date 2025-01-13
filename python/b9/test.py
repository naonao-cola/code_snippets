from ultralytics import YOLO


model = YOLO(f"yolo11s")  # build a new model from scratch
model.train(data = "/home/zhangx/project/B9/dataset_B.yaml",
            cache = False,
            imgsz = (1280, 720),
            epochs = 400,
            batch = 16,
            close_mosaic = 50,
            workers = 8,
            device = 0,
            # optimizer='SGD', # using SGD
            # amp=False, # close amp
            project=f'runs',
            name="test",
            patience=1000,
            )