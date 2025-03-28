import sys
sys.path.append(".")

from engine.core.infer import ModelWrapper
import cv2
import numpy as np
from tools.with_draw import draw_boxes


class YOLOv8Wrapper(ModelWrapper):
    def __init__(self, model_path, confidence, target_size):
        super().__init__(model_path)
        self.conf_thres = confidence
        self.target_size = target_size
    

    def preprocess(self, ori_image):
        self.scale = (ori_image.shape[0] / 640, ori_image.shape[1] / 640)  # scale = (h, w)
        ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        im_resized = cv2.resize(ori_image, self.target_size)
        im_data = im_resized.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        im_data = np.transpose(im_data, (2, 0, 1))
        im_data = np.expand_dims(im_data, axis=0)  # shape: (1, 3, *target_size)
        return im_data,
        

    def postprocess(self, model_output):

        outputs = np.squeeze(model_output[0]).T
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.conf_thres:
                box = [
                    outputs[i][0] - (0.5 * outputs[i][2]),
                    outputs[i][1] - (0.5 * outputs[i][3]),
                    outputs[i][2],
                    outputs[i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        res = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            x = round(box[0] * self.scale[1])
            y = round(box[1] * self.scale[0])
            w = round(box[2] * self.scale[1])
            h = round(box[3] * self.scale[0])
            res.append((x, y, w, h, int(class_ids[index]), float(scores[index])))
        return res


if __name__ == "__main__":
    # model_path = "model_zoo/model_files/onnx/yolov8n.onnx"
    model_path = "model_zoo/model_files/engine/yolov8n.engine"

    import os
    jgp_files = [os.path.join("dataset/coco_test", i) for i in os.listdir("dataset/coco_test")]

    for jpg_f in jgp_files:
        img = cv2.imread(jpg_f)
        model = YOLOv8Wrapper(model_path, confidence=0.25, target_size=(640, 640))
        res = model.run(img)
        print(res)


        draw_image = draw_boxes(img, res)
        cv2.imshow("Output", draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()