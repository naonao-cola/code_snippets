import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image as PILImage




def preprocess_image(image_path, input_size=(1024, 1024)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0
    image = (image-0.5) / 0.5
    img_t = np.expand_dims(image, 0)
    img_t = np.transpose(img_t, (0, 3, 1, 2))
    return img_t


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def get_pseudo_color_map(pred, color_map=None, use_multilabel=False):
    if use_multilabel:
        bg_pred = (pred.sum(axis=0, keepdims=True) == 0).astype('int32')
        pred = np.concatenate([bg_pred, pred], axis=0)
        gray_idx = np.arange(pred.shape[0]).astype(np.uint8)
        pred = (pred * gray_idx[:, None, None]).sum(axis=0)
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def _save_imgs(results, imgs_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, result in enumerate(results):
        if result.ndim == 3:
            result = np.squeeze(result)
        result = get_pseudo_color_map(result)
        basename = os.path.basename(imgs_path[i])
        basename, _ = os.path.splitext(basename)
        basename = f'{basename}.png'
        save_path = os.path.join(save_dir, basename)
        result.save(save_path)
        print('Predicted image is saved in {}'.format(save_path))


if __name__ == '__main__':

    # 模型文件
    model_file = "/home/proj/www/py/model/onnx/SPHERICAL_FOCAL.onnx"
    img_file = "/home/proj/www/py/data/paddle/3_1.bmp"
    ort_session = ort.InferenceSession(
        model_file, providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
    preprocess_image(img_file)

    # 推理
    img_input = preprocess_image(img_file)
    output = ort_session.run(output_names=None, input_feed={
                             ort_session.get_inputs()[0].name: img_input})
    # 官方代码
    _save_imgs(output, [img_file], "./output")


    # 下列保存的是黑白图像
    img_gray = output[0]
    # 去除一个维度，很重要，不然结果无法保存，只能删除单维度
    img_gray = np.squeeze(img_gray)
    img_gray = (img_gray * 255).astype(np.uint8)
    img_gray = np.clip(img_gray, 0, 255)
    output_path = "./output/segmentation_result.jpg"
    cv2.imshow("result", img_gray)
    cv2.waitKey()
    cv2.imwrite(output_path, img_gray)
    print(f"分割结果已保存到: {output_path}")
