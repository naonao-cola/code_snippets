import glob
import os

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# 图像路径
image_path = '/data/hjx/B19/data/TVB/train_val/1120'
output_path = '/data/hjx/B19/data/TVB/train_val/2'

if not os.path.exists(output_path):
    os.makedirs(output_path)

img_files = glob.glob(image_path + '/*.jpg')
for img_file in img_files:
    label = []
    xml_file = img_file.replace('.jpg', '.xml')
    tree = ET.parse(xml_file)
    myroot = tree.getroot()
    for obj in myroot.iter("object"):
        for bndbox in obj.findall("bndbox"):
            label = [int(bndbox[0].text), int(bndbox[1].text), int(bndbox[2].text), int(bndbox[3].text)]

    # 读取图像
    img = cv2.imread(img_file)

    # 检查图像是否正确加载
    if img is None:
        print("Error: Unable to load image.")
    else:
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(output_path, 'gray.jpg'), gray)

        # 使用高斯滤波器对图像进行平滑处理
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用 Canny 算子检测边缘
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        cv2.imwrite(os.path.join(output_path, 'canny.jpg'), edges)

        kernel = np.ones((2, 2), np.uint8)

        # 膨胀边缘
        dilation = cv2.dilate(edges, kernel, iterations=1)
        cut_img = dilation[label[1]:label[3], label[0]:label[2]]
        # cv2.imwrite(os.path.join(output_path, 'cut_img.jpg'), cut_img)

        # 使用OTSU方法计算最佳阈值
        contours, _ = cv2.findContours(cut_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            contour_areas.append(area)

        # 提取轮廓坐标，并向下移动20像素
        moved_contours = []
        contour_areas = []
        for contour in contours:
            moved_contour = contour + [label[0], label[1]]
            moved_contours.append(moved_contour)
            #计算面积
            area = cv2.contourArea(contour)
            contour_areas.append(area)

        index = contour_areas.index(max(contour_areas))

        cv2.drawContours(img, [moved_contours[index]], 0, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(output_path, f'{img_file.split("/")[-1].split(".")[0]}_counters.jpg'), img)
        print(max(contour_areas))