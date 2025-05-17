import os
import cv2
import numpy as np
############
# 该脚本用于保存指定目录下图像路径
###########


def save_img_path(img_dir, save_txt_path, merge_rate):
    with open(save_txt_path, "w") as f:
        for filename in os.listdir(img_dir):
            rand_num = np.random.uniform(0, 1)
            if rand_num<merge_rate:
                img_path = os.path.join(img_dir, filename)
                f.writelines(img_path+"\n")


if __name__ == '__main__':
    img_dir = r"/home/proj/www/py/data/after"  # 需处理的图像目录
    save_txt_path = r"/home/proj/www/py/data/txt/CLARITY_COARSE.txt"  # 图像目录保存文件路径
    merge_rate = 1
    save_img_path(img_dir, save_txt_path, merge_rate)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
