import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = '../onnx_model/INCLINE_RBC.onnx'
RKNN_MODEL = '../rknn_model/INCLINE_RBC.rknn'
IMG_PATH = './images/20230705-144351.468878-07.bmp'
DATASET = './datasets/incline_rbc.txt'
QUANTIZE_ON = True
conf = 0.5
OBJ_THRESH = 0.55
NMS_THRESH = 0.45
IMG_SIZE = 1024


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
    ret = rknn.hybrid_quantization_step2(model_input='./INCLINE_RBC.model',
                                         data_input="./INCLINE_RBC.data",
                                         model_quantization_cfg='INCLINE_RBC.quantization.cfg')
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
    pred_onx = rknn.inference(inputs=[img], data_format='nchw')
    pred_poly = pred_onx[0]
    pred_score = pred_onx[1]
    score_than_conf = pred_score > conf

    for chosed_poly_idx in np.nonzero(score_than_conf)[-1]:
        chosed_poly = pred_onx[0][:, chosed_poly_idx, :]

        chosed_poly = make_poly_points(chosed_poly)
        print(chosed_poly)
        cv2.polylines(img_origin, [chosed_poly.astype(np.int32)], color=(0, 0, 255), isClosed=True, thickness=1)
    cv2.imwrite("./onnx_rest.png", img_origin)

rknn.release()
