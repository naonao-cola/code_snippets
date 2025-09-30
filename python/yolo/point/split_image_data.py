from sklearn.model_selection import train_test_split
import os
import shutil
import cv2
from pathlib import Path


def split_train_val(train_path_set, val_path_set):
    total_files = []

    for filename in os.listdir(train_path_set):
        total_files.append(filename)
        # test_size为训练集和测试集的比例
    train_files, val_files = train_test_split(total_files, test_size=0.1, random_state=42)
    save_dir = Path(val_path_set)
    if save_dir.is_dir():
        for j in range(len(val_files)):
            val_path1 = train_path_set + '/' + val_files[j]
            shutil.move(val_path1, val_path_set)
    else:
        os.makedirs(save_dir)
        for j in range(len(val_files)):
            val_path1 = train_path_set + '/' + val_files[j]
            shutil.move(val_path1, val_path_set)


if __name__ == '__main__':
    train_path = r'/home/greatek/wangww/datasets/point/images/train/'  # 图片路径
    val_path = r'/home/greatek/wangww/datasets/point/images/val/'  # 划分测试集存放路径
    split_train_val(train_path, val_path)
    print("划分完成！")
