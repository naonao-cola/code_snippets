import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.transforms as T
from PIL import Image

def classify_transforms(size=224):
    # Transforms to apply if albumentations not installed
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size)])

RKNN_MODEL = '../rknn_model/CLARITY_FAR_NEAR.rknn'
device_id = '420035ef8cdff1f9'
#DATASET = './dataset.txt'
#test_dir = "/media/y/Elements/rknn_test/clarity/20240122/1"
#test_dir = "/media/y/GUTAI/windows/cell/datasets/clairity/AI/real/clarity-data1.0/quantization"
test_dir = "/media/y/GUTAI/ubuntu/rknn-toolkit2-master/rknn-toolkit2/examples/onnx/yolov5/test_rknn"
save_dir = "./clarity_far_near_b23"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

QUANTIZE_ON = True

OBJ_THRESH = 0.3
NMS_THRESH = 0.45
#wh
IMG_SIZE = (320, 320)



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


def draw_cls(image,  classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    cols_nums = 20
    for i in range(classes.shape[0]):
        horizon_position = i/cols_nums+1
        vertical_position = i %cols_nums+1
        put_text_left = (int)(image.shape[1] * (1./cols_nums) * horizon_position)
        put_text_top = (int)(image.shape[0] * (1./cols_nums) * vertical_position)
        cv2.putText(image, str(i) + ": " + str(classes[i]), (put_text_left, put_text_top),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1)



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
    ret = rknn.init_runtime(target="rk3588",  perf_debug = False, device_id=device_id)
    # ret = rknn.init_runtime(target="rk3588",  perf_debug = False, device_id=device_id, eval_mem=True)
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    test_transform = classify_transforms(size=320)
    area_box = []
    for filename in os.listdir(test_dir):
        print(filename)
        img_path = os.path.join(test_dir, filename)
        save_path = os.path.join(save_dir, filename)

        # Set inputs
        img = cv2.imread(img_path)
        #img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # test effect of to tensor
        # img = test_transform(Image.fromarray(img_ori))
        # img = np.asarray(img)
        # img = img*255
        # img = img.astype(np.uint8)
        # img = np.expand_dims(img, 0)



        img = center_crop(img_ori, IMG_SIZE)
        #img = cv2.flip(img, 0)
        cv2.imwrite("./flip.bmp", img)
        img = np.asarray(img)
        img = img.astype(np.uint8)
        img = np.expand_dims(img, 0)
        img = np.transpose(img, (0, 3, 1, 2))

        # Inference
        outputs = rknn.inference(inputs=[img], data_format='nchw')
        outputs = np.asarray(outputs)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        draw_cls(img_ori, outputs.squeeze())
        cv2.imwrite(save_path, img_ori)


        # rknn.eval_memory(
        #     is_print=True,
        # )

    rknn.release()
