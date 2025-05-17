import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '../onnx_model/CLARITY_FAR_NEAR.onnx'
RKNN_MODEL = '../rknn_model/CLARITY_FAR_NEAR.rknn'
IMG_PATH = '/media/y/GUTAI/ubuntu/rknn-toolkit2-master/rknn-toolkit2/examples/onnx/yolov5/images/img_g0c1n0001_MF0007v08(3.00).bmp'
DATASET = './datasets/clarity_normal_5_resize.txt'
QUANTIZE_ON = True
conf = 0.5
OBJ_THRESH = 0.55
NMS_THRESH = 0.45
IMG_SIZE = 320


#[1,8]->[1,4,2]
def make_poly_points(pred_points):
    poly_points = []
    for i in range(int(len(pred_points[0])/2)):
        poly_points.append([pred_points[0,2*i], pred_points[0,2*i+1]])
    return np.array(poly_points)

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)
    # Build model
    print('--> hybrid_quantization_step2')
    ret = rknn.hybrid_quantization_step2(model_input='./CLARITY_FAR_NEAR.model',
                                         data_input="./CLARITY_FAR_NEAR.data",
                                         model_quantization_cfg='CLARITY_FAR_NEAR.quantization.cfg')
    if ret != 0:
        print('hybrid_quantization_step2 failed!')
        exit(ret)
    print('done')


    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=[IMG_PATH], target="rk3588")
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')


    # Set inputs
    img_origin = cv2.imread(IMG_PATH)
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    img = cv2.resize(img, (1024, 1024))
    # img = (np.array(img) / 255.0).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))

    # Inference
    print('--> Running model')

    img_path =IMG_PATH
    # Set inputs
    img_origin = cv2.imread(img_path)
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :320, :]
    img = np.asarray(img)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))

    # Inference
    print('--> Running model')
    pred_onx = rknn.inference(inputs=[img], data_format='nchw')
    print(pred_onx)
    # print(softmax(pred_onx))

rknn.release()
