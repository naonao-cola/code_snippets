import argparse

import cv2
import numpy as np
import onnxruntime as ort



class DEIM:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.5, num_classes=9):
        self.input_width = 640
        self.input_height = 640

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.model_input = self.model.get_inputs()

        self.classes = list(range(num_classes))
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _  = img.shape
        orig_size = np.array([w, h])[None]
        im_resized = cv2.resize(img, (640, 640))
        im_data = im_resized.astype(np.float32) / 255.0  # 归一化到 [0, 1] 范围
        im_data = np.transpose(im_data, (2, 0, 1))
        im_data = np.expand_dims(im_data, axis=0)  # shape: (1, 3, 640, 640)
        
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

    def run(self, img):
        img_data, orig_target_sizes = self.preprocess(img)

        outputs = self.model.run(None, {self.model.get_inputs()[0].name: img_data,
                                        self.model.get_inputs()[1].name: orig_target_sizes})
        res = self.postprocess(outputs)  # output image

        return res


def draw_boxes(image, labels, class_names=None):
    for label in labels:
        x, y, w, h, cls, conf = label
        x, y, w, h = int(x), int(y), int(w), int(h)
        x2, y2 = x + w, y + h
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        if class_names and cls in class_names:
            label_text = f"{class_names[cls]} {conf:.2f}"
        else:
            label_text = f"Class {cls} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), -1)
        cv2.putText(image, label_text, (x, y - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image

if __name__ == "__main__":
    model_path = "models_onnx/DEIM.onnx"
    img_path = "test.jpg"


    img = cv2.imread(img_path)
    model = DEIM(model_path)
    res = model.run(img)

    print(res)

    draw_image = draw_boxes(img, res)

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", draw_image)
    cv2.waitKey(0)