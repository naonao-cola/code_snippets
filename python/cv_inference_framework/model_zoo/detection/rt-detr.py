import sys
sys.path.append(".")

from engine.core.infer import ModelWrapper
import cv2
import numpy as np
from tools.with_draw import draw_boxes

class RTDETRWrapper(ModelWrapper):
    def __init__(self, model_path, confidence, target_size):
        super().__init__(model_path)
        self.conf_thres = confidence
        self.target_size = target_size


    def preprocess(self, ori_image):
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        self.h, self.w, _  = ori_image.shape
        im_resized = cv2.resize(ori_image, self.target_size)
        im_data = im_resized.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        im_data = np.transpose(im_data, (2, 0, 1))
        im_data = np.expand_dims(im_data, axis=0)  # shape: (1, 3, *target_size)
        return im_data,
    
    
    def postprocess(self, model_output):
        outputs = np.squeeze(model_output[0])
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]
        labels = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        mask = scores > self.conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
        boxes = self.bbox_cxcywh_to_xyxy(boxes)
        
        res = []
        boxes[:, 0::2] *= self.w
        boxes[:, 1::2] *= self.h
        for i, j, k in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, i)
            res.append((x1, y1, x2-x1, y2-y1, int(k), float(j)))
        return res
    

    def bbox_cxcywh_to_xyxy(self, boxes):
        half_width = boxes[:, 2] / 2
        half_height = boxes[:, 3] / 2
        x_min = boxes[:, 0] - half_width
        y_min = boxes[:, 1] - half_height
        x_max = boxes[:, 0] + half_width
        y_max = boxes[:, 1] + half_height
        return np.column_stack((x_min, y_min, x_max, y_max))
    

if __name__ == "__main__":
    # model_path = "model_zoo/model_files/onnx/rtdetr-l.onnx"
    model_path = "model_zoo/model_files/engine/rtdetr-l.engine"

    import os
    jgp_files = [os.path.join("dataset/coco_test", i) for i in os.listdir("dataset/coco_test")]

    for jpg_f in jgp_files:
        img = cv2.imread(jpg_f)
        model = RTDETRWrapper(model_path, confidence=0.25, target_size=(640, 640))
        res = model.run(img)
        print(res)


        draw_image = draw_boxes(img, res)
        # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()