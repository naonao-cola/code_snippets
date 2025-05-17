#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pathlib import Path
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt
from PIL import Image
from utils.ulog import ulog
from typing import (
    Optional,
)
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def calculate_base_zoom(src_width, src_height, window_width, window_height) -> float:
    """_summary_
    # 计算图像的宽高与窗口宽高的最小比例
    # 计算适应窗口大小的基础缩放比例
    """
    # 计算适应窗口的缩放比例
    width_ratio = window_width / src_width
    height_ratio = window_height / src_height
    # 选择较小的比例以确保图片完全显示
    return min(width_ratio, height_ratio)


def load_image(image_path) -> QImage:
    """
    Loads an image from the specified path.
    """
    try:
        if image_path:
            image = Image.open(image_path)
            image = image.convert("RGB")
            data = image.tobytes("raw", "RGB")
            q_image = QImage(data, image.width, image.height, QImage.Format_RGB888)
        else:
            q_image = QImage(800,800,QImage.Format_RGB888)
            q_image.fill(Qt.black)
            ulog().get_logger().error(f" 不存在图片，构造空图")
        ret = QPixmap.fromImage(q_image)
        return ret
    except Exception as e:
        ulog().get_logger().error(
            f" path {image_path} Error loading image: {str(e)}")
        return None


def scale_image(image: QPixmap,
                base_zoom: float,
                zoom_factor: float,
                window_width: int,
                window_height: int,
                w_rate: Optional[float] = None,
                h_rate: Optional[float] = None):
    """
    缩放图像，
    base_zoom: 缩放基准值，默认为1.0
    zoom_factor: 缩放因子，默认为1.0
    center_x: 缩放中心x坐标(相对于窗口)
    center_y: 缩放中心y坐标(相对于窗口)
    """
    if not image:
        return
    image_width = image.width()
    image_height = image.height()

    scaled_width = int(image_width * base_zoom * zoom_factor)
    scaled_height = int(image_height * base_zoom * zoom_factor)

    # 缩放后的图
    scaled_pixmap = image.scaled(
        scaled_width,
        scaled_height,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )

    # 创建黑色背景的图片
    black_pixmap = QPixmap(window_width, window_height)
    black_pixmap.fill(Qt.black)
    painter = QPainter(black_pixmap)

    if w_rate is None or h_rate is None:
        x = (window_width - scaled_width) // 2
        y = (window_height - scaled_height) // 2
    else:
        x = - scaled_width * w_rate
        y = - scaled_height * h_rate

    painter.drawPixmap(int(x), int(y), scaled_width,
                       scaled_height, scaled_pixmap)
    painter.end()
    return scaled_pixmap, black_pixmap, scaled_width, scaled_height, x, y


def move_image(scaled_pixmap: QPixmap, black_pixmap: QPixmap, x, y, dx, dy):
    """
    x 与 y 是图像缩放的大图，在窗口上显示的左上角的坐标，
    """
    dst_image = QPixmap(black_pixmap.width(), black_pixmap.height())
    dst_image.fill(Qt.black)
    painter = QPainter(dst_image)
    painter.drawPixmap(int(x + dx), int(y+dy), scaled_pixmap.width(),
                       scaled_pixmap.height(), scaled_pixmap)
    painter.end()
    return dst_image


def normalize_path(path):
    """标准化路径，转换为绝对路径并统一分隔符"""
    return os.path.abspath(os.path.normpath(path)).lower()


def get_image_list(image_path: str, image_list: list) -> list:
    # 获取图片列表
    image_extensions = {'.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.tiff', '.webp'}
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_item_path = os.path.join(root, file)
                normal_item_path = normalize_path(image_item_path)
                if normal_item_path not in image_list:
                    image_list.append(normal_item_path)
    return image_list


def process_directory(directory, idx_len=4):
    """处理单个目录，返回符合条件的文件夹路径列表"""
    folders = []
    path = Path(directory)
    for p in path.iterdir():
        if p.is_dir():
            current_dir_name = p.stem
            if '_' in str(current_dir_name):
                parts = current_dir_name.split('_', 1)
                if len(parts) >= 2:
                    sample_name, cate_name = parts[0], parts[1]
                    if len(sample_name) == idx_len and cate_name.isalpha():
                        folders.append(str(p))
    return folders


def get_all_folders(directory, idx_len=4, max_workers=os.cpu_count()):
    """获取测试编号的目录

    Args:
        directory (str): 目录 sn 的目录
        idx_len (int, optional): 测试编号的长度. Defaults to 4.
        max_workers (int, optional): 最大进程数. Defaults to None.

    Returns:
        list: 符合条件的文件夹路径列表
    """
    root_path = Path(directory)
    if not root_path.exists() or not root_path.is_dir():
        print(f"目录不存在: {directory}")
        return []

    subdirectories = [subdir for subdir in root_path.iterdir() if subdir.is_dir()]
    if not subdirectories:
        return process_directory(directory, idx_len)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for subdir in subdirectories:
            futures.append(executor.submit(process_directory, str(subdir), idx_len))

        all_folders = []
        for future in as_completed(futures):
            try:
                result = future.result()
                all_folders.extend(result)
            except Exception as e:
                ulog().get_logger().error(f"多进程处理目录时错误: {e}")
    return all_folders


def process_image(item):
    try:
        # F:\data\SN\20250501\0001_cbc
        image_name = os.path.basename(item)  # 图片路径
        # 0001_cbc
        sample_name, cate_name = image_name.split('_', 1)
        date_name = os.path.basename(os.path.dirname(item))  # 日期
        sn_name = os.path.basename(os.path.dirname(os.path.dirname(item)))  # sn号
        return {
            "sn": sn_name,
            "date": date_name,
            "sample_id": sample_name,
            "categorize": cate_name,
            "path": item,
        }
    except Exception as e:
        print(f"Error processing {item}: {e}")
        return None


def process_image_vec(image_vec, max_workers=os.cpu_count()):
    file_data = []
    if not image_vec:
        return file_data

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_image, image_vec)

    for result in results:
        if result is not None:
            file_data.append(result)

    return file_data
