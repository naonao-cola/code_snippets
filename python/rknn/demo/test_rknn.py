import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

RKNN_MODEL = '../rknn_model/WBC4_20240627.rknn'
device_id = '420035ef8cdff1f9'
#DATASET = './dataset.txt'
test_dir = "/media/y/B2DC8AA5DC8A6407/wbc/4/val/wbc-4_quantization_resize"
save_dir = "./wbc4"

specific_anchor = [[2.345703125, 2.59765625], [3.3671875, 2.46875], [2.7578125, 3.458984375], [3.921875, 2.748046875],
     [3.560546875, 3.576171875], [3.701171875, 4.4609375], [4.53515625, 4.37890625], [5.65625, 5.77734375],
     [9.46875, 10.171875]]
specific_anchor = None
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

QUANTIZE_ON = True

OBJ_THRESH = 0.5
NMS_THRESH = 0.4
IMG_SIZE = 1920


ratio_img = 2.095833333 * 2.095833333
ratio_box_thr = 0.7
print(ratio_img)
# CLASSES = ("NEU", "LYM", "MONO", "EOS")
CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]] if specific_anchor is None else specific_anchor


    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)



def recover_box(box, img_ori_shape=[3036, 4024], resized_shape = 1920):
    in_w = img_ori_shape[1]
    in_h = img_ori_shape[0]
    tar_w = resized_shape
    tar_h = resized_shape
    r = min(float(tar_h) / in_h, float(tar_w) / in_w)
    inside_w = int(round(in_w * r))
    inside_h = int(round(in_h * r))
    padd_w = tar_w - inside_w
    padd_h = tar_h - inside_h
    top = int(round(padd_h / 2 - 0.1))
    bottom = int(round(padd_h - top + 0.1))
    left = int(round(padd_w / 2 - 0.1))
    right = int(round(padd_w - left + 0.1))
    x1, y1, x2, y2 = box
    x1 = x1 - left
    x2 = x2 - left
    y1 = y1 - top
    y2 = y2 - top
    x1 = int(x1 / r)
    x2 = int(x2 / r)
    y1 = int(y1 / r)
    y2 = int(y2 / r)
    return [x1, y1, x2, y2]


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)



    # Load ONNX model
    print('--> Loading model')
    # ret = rknn.load_onnx(model=ONNX_MODEL)
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rk3588",  perf_debug = False, device_id=device_id, eval_mem=True)
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')




    area_box = []
    for filename in os.listdir(test_dir):
        print(filename)
        img_path = os.path.join(test_dir, filename)
        save_path = os.path.join(save_dir, filename)

        # Set inputs
        img = cv2.imread(img_path)
        #img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img, ratio, (dw, dh) = letterbox(img_ori, new_shape=(IMG_SIZE, IMG_SIZE))
        img = img.astype(np.uint8)
        # Inference
        outputs = rknn.inference(inputs=[img])



        # thr_thresholds = np.arange(0.25, 0.90, 0.05)
        thr_thresholds = {OBJ_THRESH}
        for i in thr_thresholds:
            OBJ_THRESH = i
            # post process
            input0_data = outputs[0]
            input1_data = outputs[1]
            input2_data = outputs[2]

            input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
            input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
            input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

            input_data = list()
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

            boxes, classes, scores = yolov5_post_process(input_data)

            img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if boxes is not None:
                draw(img_1, boxes, scores, classes)
                cv2.imwrite(save_path, img_1)
                for box in boxes:
                    left, top,  right, bottom = box

                    if left < 0 or top < 0 or right > img_ori.shape[1] or bottom > img_ori.shape[0]:
                        print(box)
                        continue
                    height = bottom - top
                    width =right - left
                    if abs(1-height/width)<ratio_box_thr:
                        area_box.append((bottom-top)*(right-left))


        # rknn.eval_memory(
        #     is_print=True,
        # )

    mean_value = np.array(area_box).mean().item()
    std_value = np.array(area_box).std().item()
    print(mean_value)
    print(std_value)

    rknn.release()
