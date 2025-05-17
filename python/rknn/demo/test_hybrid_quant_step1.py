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

    # pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]], target_platform="rk3588")
    rknn.config(mean_values=[[124, 116, 104]], std_values=[[58, 57, 57]], target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    # ret = rknn.load_onnx(model=ONNX_MODEL, inputs=['image'], input_size_list=[[1,3,1920,1920]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset=DATASET, proposal=False)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    rknn.release()
