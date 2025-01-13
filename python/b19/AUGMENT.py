# -*- coding: utf-8 -*-
import shutil

import cv2
import numpy as np
import os.path
import copy


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


if __name__ == "__main__":
    # 图片文件夹路径
    num = 4
    file_dir = r'/data/hjx/B19/data/POAPS/POA客诉'
    out_path = r'/data/hjx/B19/data/POAPS/out'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(num):
        for img_name in os.listdir(file_dir):
            if img_name.endswith(".jpg"):
                img_path = file_dir + '/' +  img_name
                xml_path = file_dir + '/' +  img_name.replace('.jpg', '.xml')
                img = cv2.imread(img_path)
                # 镜像
                flipped_img = flip(img)
                cv2.imwrite(out_path + '/' +  img_name[0:-4] + f'_fli{i}.jpg', flipped_img)

                blur = cv2.GaussianBlur(img, (3, 3), 1.5)
                GaussianBlur_name = out_path + '/' + img_name[0:-4] + f'_GaussianBlur{i}.jpg'
                cv2.imwrite(GaussianBlur_name, blur)
                shutil.copy(xml_path, GaussianBlur_name.replace('.jpg', '.xml'))

                # 增加噪声
                # img_salt = SaltAndPepper(img, 0.01)
                # salt_name = out_path + '/' +  img_name[0:-4] + '_salt.jpg'
                # cv2.imwrite(salt_name, img_salt)
                # shutil.copy(xml_path, salt_name.replace('.jpg', '.xml'))
                #
                # img_gauss = addGaussianNoise(img, 0.01)
                # gauss_name = out_path + '/' +  img_name[0:-4] + f'_noise{i}.jpg'
                # cv2.imwrite(gauss_name, img_gauss)
                # shutil.copy(xml_path, gauss_name.replace('.jpg', '.xml'))

                # 变亮、变暗
                img_darker = darker(img)
                darker_name = out_path + '/' +  img_name[0:-4] + f'_darker{i}.jpg'
                cv2.imwrite(darker_name, img_darker)
                shutil.copy(xml_path, darker_name.replace('.jpg', '.xml'))

                img_brighter = brighter(img)
                brighter_name = out_path + '/' +  img_name[0:-4] + f'_brighter{i}.jpg'
                cv2.imwrite(brighter_name, img_brighter)
                shutil.copy(xml_path, brighter_name.replace('.jpg', '.xml'))

