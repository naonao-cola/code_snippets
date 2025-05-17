import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

from collections import namedtuple
ONNX_MODEL = '../onnx_model/SPHERICAL_FOCAL.onnx'
RKNN_MODEL = '../rknn_model/SPHERICAL_FOCAL.rknn'
IMG_PATH = './images/20240428-130851.372158.bmp'
DATASET = './datasets/PPSEG.txt'
#DATASET = './dataset.txt'
QUANTIZE_ON = True
conf = 1
OBJ_THRESH = 0.55
NMS_THRESH = 0.45
IMG_SIZE = 1024

Cls = namedtuple('cls', ['name', 'id', 'color'])

Clss = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Cls(  'unlabeled'            ,  0 ,       (  0,  0,  0) ),
    # Cls(  'ego vehicle'          ,  1 ,       (  0,  0,  0) ),
    Cls(  'ego vehicle'          ,  1 ,       (  255,  255,  255) ),
    Cls(  'rectification border' ,  2 ,      (  0,  0,  0) ),
    Cls(  'out of roi'           ,  3 ,       (  0,  0,  0) ),
    Cls(  'static'               ,  4 ,      (  0,  0,  0) ),
    Cls(  'dynamic'              ,  5 ,       (111, 74,  0) ),
    Cls(  'ground'               ,  6 ,       ( 81,  0, 81) ),
    Cls(  'road'                 ,  7 ,      (128, 64,128) ),
    Cls(  'sidewalk'             ,  8 ,      (244, 35,232) ),
    Cls(  'parking'              ,  9 ,      (250,170,160) ),
    Cls(  'rail track'           , 10 ,       (230,150,140) ),
    Cls(  'building'             , 11 ,       ( 70, 70, 70) ),
    Cls(  'wall'                 , 12 ,       (102,102,156) ),
    Cls(  'fence'                , 13 ,       (190,153,153) ),
    Cls(  'guard rail'           , 14 ,        (180,165,180) ),
    Cls(  'bridge'               , 15 ,       (150,100,100) ),
    Cls(  'tunnel'               , 16 ,       (150,120, 90) ),
    Cls(  'pole'                 , 17 ,       (153,153,153) ),
    Cls(  'polegroup'            , 18 ,        (153,153,153) ),
    Cls(  'traffic light'        , 19 ,        (250,170, 30) ),
    Cls(  'traffic sign'         , 20 ,      (220,220,  0) ),
    Cls(  'vegetation'           , 21 ,      (107,142, 35) ),
    Cls(  'terrain'              , 22 ,      (152,251,152) ),
    Cls(  'sky'                  , 23 ,      ( 70,130,180) ),
    Cls(  'person'               , 24 ,      (220, 20, 60) ),
    Cls(  'rider'                , 25 ,       (255,  0,  0) ),
    Cls(  'car'                  , 26 ,       (  0,  0,142) ),
    Cls(  'truck'                , 27 ,       (  0,  0, 70) ),
    Cls(  'bus'                  , 28 ,        (  0, 60,100) ),
    Cls(  'caravan'              , 29 ,       (  0,  0, 90) ),
    Cls(  'trailer'              , 30 ,        (  0,  0,110) ),
    Cls(  'train'                , 31 ,       (  0, 80,100) ),
    Cls(  'motorcycle'           , 32 ,       (  0,  0,230) ),
    Cls(  'bicycle'              , 33 ,       (119, 11, 32) ),
    Cls(  'license plate'        , -1 ,       (  0,  0,142) ),
]


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

def gray_color(color_dict, img):
    '''
    swift gray image to color, by color mapping relationship
    :param color_dict:color mapping relationship, dict format
    :param gray_path:gray imgs path
    :param color_path:color imgs path
    :return:
    '''

    # print(gt_gray)
    assert len(img.shape) == 2  # make sure gt_gray is 1band

    gt_color = matrix_mapping(color_dict, img)
    # endregion

    gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
    return gt_color

def matrix_mapping(color_dict, gt):
    colorize = np.zeros([len(color_dict), 3], 'uint8')
    for cls, color in color_dict.items():
        colorize[cls, :] = list(color)
    ims = colorize[gt, :]
    ims = ims.reshape([gt.shape[0], gt.shape[1], 3])
    return ims

#[1,8]->[1,4,2]
def make_poly_points(pred_points):
    poly_points = []
    for i in range(int(len(pred_points[0])/2)):
        poly_points.append([pred_points[0,2*i], pred_points[0,2*i+1]])
    return np.array(poly_points)

def nt_dic(nt=Clss):
    '''
    swift nametuple to color dict
    :param nt: nametuple
    :return:
    '''
    pass
    color_dict = {}
    for cls in nt:
        color_dict[cls.id] = cls.color
    return color_dict

if __name__ == '__main__':
    color_dict = nt_dic()


    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]], target_platform="rk3588", optimization_level=2)
    rknn.config(mean_values=[[127, 127, 127]], std_values=[[127, 127, 127]], target_platform="rk3588", optimization_level=2)
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
    # ret = rknn.accuracy_analysis(inputs=[IMG_PATH], target=None)
    # if ret != 0:
    #     print('Accuracy analysis failed!')
    #     exit(ret)
    # print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img_origin = cv2.imread(IMG_PATH)

    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)


    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img, ratio, _ = letterbox(img, (IMG_SIZE, IMG_SIZE))

    img_t = np.expand_dims(img, 0)
    img_t = np.transpose(img_t, (0, 3, 1, 2))
    # Inference
    print('--> Running model')
    pred_onx = rknn.inference(inputs=[img_t], data_format='nchw')


    img_gray = pred_onx[0][0]

    img_gray = img_gray[1] > img_gray[0]
    # img_gray = img_gray.squeeze()

    # img_save = cv2.resize(img_origin, (IMG_SIZE, IMG_SIZE))
    img_save = img
    img_save[:, :, 2] = img_save[:, :, 2]/2 + img_gray*255/2
    cv2.imwrite("./seg_result_20231130-160145.8333700.png", img_save.astype(np.uint8))
    cv2.imwrite("./resized.png", img)


    rknn.release()
