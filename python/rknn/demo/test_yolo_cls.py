import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '../onnx_model/CLARITY_COARSE.onnx'  # 待量化onnx模型路径
RKNN_MODEL = '../rknn_model/CLARITY_COARSE.rknn'  # 量化后rknn模型保存路径
IMG_PATH = ['./images/000015.bmp']  # 量化后测试图像
DATASET = './datasets/CLARITY_COARSE.txt'  # 保存有量化图像绝对路径的txt文件

QUANTIZE_ON = True
conf = 0.5
OBJ_THRESH = 0.55
NMS_THRESH = 0.45

#w,h
#NORMAL
IMG_SIZE = (320, 320)  # yolo cls采用保持纵横比的方式resize图像后,再进行裁减得道输入图像.于文件99行进行裁减.

#BASO
#IMG_SIZE = (320, 320)

CLASSES = ("0", "1", "2", "3", "4", "5", "6")

device_id = '420035ef8cdff1f9'


def center_crop(img, target_size):
    height, width = img.shape[:-1]
    resize_max_ratio = np.max([target_size[0]/np.array([height, width]), target_size[1]/np.array([height, width])] )
    inside_h = int((height*resize_max_ratio).round())
    inside_w = int((width*resize_max_ratio).round())

    img = cv2.resize(img, (inside_w, inside_h), interpolation=cv2.INTER_LINEAR)
    start_x = int((inside_h-target_size[0])/2)
    start_y = int((inside_w-target_size[1])/2)

    img = img[start_x:start_x+target_size[0], start_y:start_y+target_size[1], :]  # 裁减
    return img


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]], target_platform="rk3588")
    # rknn.config(mean_values=[[124, 116, 104]], std_values=[[58, 57, 57]], target_platform="rk3588", optimization_level=3)
    rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 58.395, 58.395]], target_platform="rk3588",
                optimization_level=3)
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
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    # print('--> Accuracy analysis')
    # ret = rknn.accuracy_analysis(inputs=["/media/y/GUTAI/windows/cell/datasets/clairity/AI/real/clarity-5.0/splited/1/img240004141503_g0c0n0001_AF0000v09(0.21).bmp"], target="rk3588", device_id=device_id)
    # if ret != 0:
    #     print('Accuracy analysis failed!')
    #     exit(ret)
    # print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    for img_path in IMG_PATH:

        # Set inputs
        img_origin = cv2.imread(img_path)
        img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
        img = center_crop(img, IMG_SIZE)
        img = np.asarray(img)
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0, 3, 1, 2))

        # Inference
        print('--> Running model')
        pred_onx = rknn.inference(inputs=[img], data_format='nchw')
        print(pred_onx)
        # print(softmax(pred_onx))

    rknn.release()

