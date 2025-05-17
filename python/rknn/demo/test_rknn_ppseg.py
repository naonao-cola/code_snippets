import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
###########################
# 当前模型未进行量化,因此并不需要DATASET字段
###########################

ONNX_MODEL = '../onnx_model/SPHERICAL_FOCAL.onnx'
RKNN_MODEL = '../rknn_model/SPHERICAL_FOCAL.rknn'
IMG_PATH = './images/20240428-130851.372158.bmp'


QUANTIZE_ON = True
conf = 0.5
OBJ_THRESH = 0.55
NMS_THRESH = 0.45
IMG_SIZE = 1024
device_id = '420035ef8cdff1f9'

CLASSES = ("0", "1")
# CLASSES = ("NEU", "LYM", "MONO", "EOS", "IG", "UNKNOWN")
#CLASSES = ("PLT")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#[1,8]->[1,4,2]
def make_poly_points(pred_points):
    poly_points = []
    for i in range(int(len(pred_points[0])/2)):
        poly_points.append([pred_points[0,2*i], pred_points[0,2*i+1]])
    return np.array(poly_points)



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



if __name__ == '__main__':

    rknn = RKNN(verbose=True)

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')



    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target="rk3588", device_id=device_id)
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img_origin = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    img, ratio, _ = letterbox(img, (IMG_SIZE, IMG_SIZE))

    #img = img[-IMG_SIZE:, -IMG_SIZE:, :]
    img_t = np.expand_dims(img, 0)
    img_t = np.transpose(img_t, (0, 3, 1, 2))


    # Inference
    print('--> Running model')
    pred_onx = rknn.inference(inputs=[img_t], data_format='nchw')
    img_gray = pred_onx[0][0]

    img_gray = img_gray[1] > img_gray[0]
    img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img_save = img_origin[-IMG_SIZE:, -IMG_SIZE:, :]
    img_save[:, :, 2] = img_save[:, :, 2]/2 + img_gray*255/2
    cv2.imwrite("./seg_result_rknn_20230704-171755.0608170_erap.png", img_save.astype(np.uint8))
    cv2.imwrite("./resized.png", img_save)

    rknn.release()
