# -*- coding: utf-8 -*-
from PIL import Image
import albumentations as A


import numpy as np
import cv2
import os
from pathlib import Path
import random
import glob
import json


def fix_jpg():
    img_file = glob.glob(
        "F:/download/sbg/IL-0701/il_all_image/right/*.JPG")
    for item in img_file:
        before_path = Path(item)
        after_path = before_path.with_suffix('.jpg')
        before_path.rename(after_path)


def transform_func():
    transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.OneOf([A.Perspective(scale=(0.05, 0.15)),
                 A.ShiftScaleRotate(shift_limit=0.0625,
                                    scale_limit=0.2, rotate_limit=10, p=0.4),
                 A.PiecewiseAffine(),
                 ],
                p=0.2),
        A.OneOf([A.GaussNoise(),], p=0.2),
        A.OneOf([A.MotionBlur(p=0.2), A.MedianBlur(
            blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1),], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625,
                           scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([A.RandomBrightnessContrast(
            brightness_limit=(0, 0.3), contrast_limit=(0, 0.3),),], p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc'))

    return transform


def transform_img(image_path, json_path):

    image = cv2.imread(image_path)
    if image is None:
        print(image_path)
        return None
    with open(json_path, 'r', encoding='utf-8') as path_json:
        jsonx = json.load(path_json)

    shapes = jsonx['shapes']
    # 获取图片长和宽
    width = jsonx['imageWidth']
    height = jsonx['imageHeight']
    bboxes = []
    for shape in shapes:
        # 获取矩形框两个角点坐标
        x1 = shape['points'][0][0]
        y1 = shape['points'][0][1]
        x2 = shape['points'][1][0]
        y2 = shape['points'][1][1]
        cat = shape['label']
        bboxes.append([x1, y1, x2, y2])

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    random.seed(42)
    trans_func = transform_func()
    augmented_ret = trans_func(image=image, bboxes=bboxes)

    image_ret = augmented_ret['image']
    bboxes_ret = augmented_ret['bboxes']
    return image_ret, bboxes_ret


def get_file(file_path: str, suffix: str, res_file_path: list) -> list:
    for file in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, file)):
            get_file(os.path.join(file_path, file), suffix, res_file_path)
        elif file.endswith(suffix):
            res_file_path.append(os.path.join(file_path, file))


def exec_augmentation(file_path):
    img_files = []
    get_file(file_path, "jpg", img_files)
    for item in img_files:
        img_path = item
        json_path = Path(img_path).with_suffix(".json").__str__()

        aug_img, aug_box = transform_img(img_path, json_path)

        example_path = Path(item)
        rand_num = random.randint(1, 500)
        img_name = file_path+"/" + \
            str(example_path.stem) + "_aug"+str(rand_num)+".JPG"
        json_name = file_path+"/" + \
            str(example_path.stem) + "_aug"+str(rand_num)+".json"
        cv2.imwrite(img_name, aug_img)


def paste_img(large_img_path: str, small_img_path: str, save_path: str):

    background_image = Image.open(large_img_path)
    foreground_image = Image.open(small_img_path)
    background_width, background_height = background_image.size
    foreground_width, foreground_height = foreground_image.size
    max_x = background_width - foreground_width
    max_y = background_height - foreground_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    background_image.paste(foreground_image, (x, y))
    background_image.save(save_path)


def exec_paste_img(big_img_path, small_img_path):

    big_img_path_list = []
    small_img_path_list = []
    get_file(big_img_path, "JPG", big_img_path_list)
    get_file(small_img_path, "png", small_img_path_list)
    max_big_len = len(big_img_path_list)
    max_small_len = len(small_img_path_list)
    for i in range(max_big_len):
        select_small_img_index = random.randint(0, max_small_len - 1)
        select_small_img = small_img_path_list[select_small_img_index]
        paste_img(big_img_path_list[i], select_small_img,
                  big_img_path_list[i][:-4] + "_" + str(select_small_img_index)+".jpg")


if __name__ == '__main__':
    exec_augmentation(
        r'Y:\proj\www\repo\yolo8_test\dataset\sbg_il_test\result')

    # exec_augmentation(r'F:\download\sbg\Tuling\new\pannel')

    # exec_paste_img(r'F:\download\sbg\Tuling\new\pannel',
    #                r'E:\demo\cxx\tv_algo_sbg\label_img\small')
    # fix_jpg()
    print("well done! ")
