import sys
sys.path.append(".")

from engine.core.infer import ModelWrapper
import cv2
import numpy as np
from tools.with_draw import draw_boxes

class DEIMWrapper(ModelWrapper):
    def __init__(self, model_path, confidence, target_size):
        super().__init__(model_path)
        self.conf_thres = confidence
        self.target_size = target_size
    

    def preprocess(self, ori_image):
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        h, w, _  = ori_image.shape
        orig_size = np.array([w, h])[None]
        im_resized = cv2.resize(ori_image, self.target_size)
        im_data = im_resized.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        im_data = np.transpose(im_data, (2, 0, 1))
        im_data = np.expand_dims(im_data, axis=0)  # shape: (1, 3, *target_size)
        return im_data, orig_size
        
        
    def postprocess(self, model_output):
        labels = model_output[0]
        boxes = model_output[1]
        scores = model_output[2]

        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        res = []
        for i, j, k in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, i)
            res.append((x1, y1, x2-x1, y2-y1, int(k), float(j)))
        return res
    

if __name__ == "__main__":
    # model_path = "model_zoo/model_files/onnx/DEIM_s.onnx"
    model_path = "model_zoo/model_files/engine/DEIM_s.engine"

    import os
    jgp_files = [os.path.join("dataset/coco_test", i) for i in os.listdir("dataset/coco_test")]

    for jpg_f in jgp_files:
        img = cv2.imread(jpg_f)
        model = DEIMWrapper(model_path, confidence=0.25, target_size=(640, 640))
        res = model.run(img)
        print(res)


        draw_image = draw_boxes(img, res)
        # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()