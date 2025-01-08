import cv2
import glob
from pathlib import Path
import numpy as np
import sys
import os


def get_all_file(file_path, end_str):
    search_path = file_path + '/**/*.' + end_str
    file_list = glob.glob(search_path, recursive=True)
    double_end_str = "(2)." + end_str
    first_path = []
    second_path = []
    for img_path_item in file_list:
        if img_path_item.endswith(double_end_str):
            second_path.append(img_path_item)
        else:
            first_path.append(img_path_item)
    first_path.sort()
    second_path.sort()
    return first_path, second_path


def hf_rename(file_path, end_str):
    first_path, second_path = get_all_file(file_path, end_str)
    count = 1
    if len(first_path) != len(second_path):
        print("img count is error")
        return
    for item1, item2 in zip(first_path, second_path):
        # 中文路径问题
        # 比较亮度，亮的作为A
        img1 = cv2.imdecode(np.fromfile(
            os.path.join(item1), dtype=np.uint8), 0)
        img2 = cv2.imdecode(np.fromfile(
            os.path.join(item2), dtype=np.uint8), 0)
        mean_brightness1 = np.mean(img1)
        mean_brightness2 = np.mean(img2)
        path1 = ""
        path2 = ""
        if mean_brightness1 > mean_brightness2:
            path1 = Path(item1)
            path2 = Path(item2)
        else:
            path1 = Path(item2)
            path2 = Path(item1)
        item1_new = str(count)+"_A.PNG"
        item2_new = str(count)+"_B.PNG"
        path1_new = path1.with_name(item1_new)
        path2_new = path2.with_name(item2_new)
        path1.rename(path1_new)
        path2.rename(path2_new)
        count += 1
    return


def hf_rename_2(file_path, end_str):
    file = glob.glob(file_path + '/**/*.' + end_str, recursive=True)
    count=0
    for item1 in file:
        path1 = Path(item1)
        item1_new = str(count)+".PNG"
        path1_new = path1.with_name(item1_new)
        path1.rename(path1_new)
        count += 1
    return


def main():
    # if len(sys.argv) < 2:
    #     print("请输入参数")
    #     return
    # input_path = sys.argv[1]
    # normalized_path = os.path.normpath(input_path)

    # hf_rename(normalized_path, sys.argv[2])
    hf_rename(r"F:\download\hf\0815bk\all_img", 'PNG')


# 第一个参数路径，第二个参数文件后缀
if __name__ == '__main__':
    # main()
    hf_rename_2(r"Y:\proj\www\hf\lc", 'PNG')
    print("well done! ")
