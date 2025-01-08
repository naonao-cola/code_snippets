import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# 读取图片
base_file = R"F:\data\echint\Oxida&Contamination"

img_path = "AD2300058_1_8622-M1_STRIP_AOI_BAC03_2_50_Oxida&Contamination_214.744_18.571_0002.png"

img_file = os.path.join(base_file, img_path)


img_bgr = cv2.imread(img_file, cv2.IMREAD_COLOR)
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
# cv2.namedWindow("input",cv2.WINDOW_GUI_NORMAL)
# cv2.imshow("input",img_lab)

# 分别获取三个通道的ndarray数据
img_ls = img_lab[:, :, 0]
img_as = img_lab[:, :, 1]
img_bs = img_lab[:, :, 2]
cv2.namedWindow("input_l", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("input_l", img_ls)

cv2.namedWindow("input_as", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("input_as", img_as)

cv2.namedWindow("input_bs", cv2.WINDOW_GUI_NORMAL)
cv2.imshow("input_bs", img_bs)


'''按L、A、B三个通道分别计算颜色直方图'''
ls_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 255])
as_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 255])
bs_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 255])
# m,dev = cv2.meanStdDev(img_lab)  #计算L、A、B三通道的均值和方差
# print(m)

'''显示三个通道的颜色直方图'''
plt.plot(ls_hist, label='l', color='blue')
plt.plot(as_hist, label='a', color='green')
plt.plot(bs_hist, label='b', color='red')
plt.legend(loc='best')
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)
