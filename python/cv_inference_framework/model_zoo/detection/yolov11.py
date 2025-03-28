import sys
sys.path.append(".")

from engine.core.infer import ModelWrapper
import cv2
import numpy as np
from tools.with_draw import draw_boxes

from yolov8 import YOLOv8Wrapper as YOLOv11Warpper



if __name__ == "__main__":
    # model_path = "model_zoo/model_files/onnx/yolo11n.onnx"
    model_path = "model_zoo/model_files/engine/yolo11n.engine"

    import os
    jgp_files = [os.path.join("dataset/coco_test", i) for i in os.listdir("dataset/coco_test")]

    for jpg_f in jgp_files:
        img = cv2.imread(jpg_f)
        model = YOLOv11Warpper(model_path, confidence=0.25, target_size=(640, 640))
        res = model.run(img)
        print(res)


        draw_image = draw_boxes(img, res)
        cv2.imshow("Output", draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()