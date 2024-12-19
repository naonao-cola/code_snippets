import glob
import os

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# 图像路径
image_path = '/data/hjx/B19/data/TVPS/1'
output_path = '/data/hjx/B19/data/TVPS/2'

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

        row_num = np.int(np.ceil((label[2] - label[0]) / 20))
        line_num = np.int(np.ceil((label[3] - label[1]) / 20))

        temp_counter = moved_contours[index].copy()
        mask = np.zeros_like(img)
        temp = mask

        # 将轮廓区域内的像素值设置为255
        cv2.drawContours(mask, [moved_contours[index]], 0, (255, 255, 255), thickness=cv2.FILLED)

        # 使用掩模将原图中对应区域的像素值设置为255
        result = np.where(mask == 255, 255, temp)
        cv2.imwrite(os.path.join(output_path, 'result.jpg'), result)

        print(label[2]-label[0], label[3]-label[1])

        temp_imglist = []
        for i, x in enumerate(range(row_num)):
            for j, y in enumerate(range(line_num)):
                x1 = label[0] + 20 * i
                x2 = x1 + 20
                y1 = label[1] + 20 * j
                y2 = y1 + 20
                cv2.rectangle(result, (x1,y1), (x2, y2), (0, 255, 0))
                cv2.imwrite(os.path.join(output_path, 'rectangle.jpg'), result)

                temp_img = result[y1:y2, x1:x2]
                temp_imglist.append(temp_img)

        for image in temp_imglist:
            white_pixel = np.array([255, 255, 255])
            non_white_mask = np.any(image == white_pixel, axis=2)
            non_white_count = np.sum(non_white_mask)
            print(non_white_count)
