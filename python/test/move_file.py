import cv2
import glob
from pathlib import Path
import numpy as np
import sys
import os
import shutil


def get_all_file(file_path, in_str):
    search_path = file_path + '/**/*.*'
    file_list = glob.glob(search_path, recursive=True)
    move_file_list = []
    for img_path_item in file_list:
        path1 = Path(img_path_item)
        if in_str in path1.name and path1.is_file():
            move_file_list.append(img_path_item)
    return move_file_list


def move(file_path, in_str, dst_path):
    move_file_list = get_all_file(r'F:\download\hf\lc\7.17', "OpenView")
    for item in move_file_list:
        print(item + "\n")

    for item in move_file_list:
        cur_path = Path(item)
        folder_to_move = cur_path.parent
        # 获取相对路径
        new_folder_path = dst_path / folder_to_move.relative_to(file_path)
        shutil.copytree(str(folder_to_move), str(
            new_folder_path), dirs_exist_ok=True)
        print(f"Moved {folder_to_move} to {new_folder_path}" + "\n")


def main():
    move(r'F:\download\hf\lc\7.17', "OpenView", r'F:\download\hf\lc\新建文件夹')


if __name__ == '__main__':
    main()
    print("well done! ")
