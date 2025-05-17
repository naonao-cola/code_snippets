#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from PIL import Image
from utils.ulog import ulog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent, QWheelEvent
from PyQt5.QtCore import Qt, QPoint, QSize, QRectF
from PyQt5.QtWidgets import (
    QLabel, QScrollArea, QSizePolicy, QMessageBox
)
from utils.image_func import *


class UImage(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # 初始化变量
        self.zoom_factor = 1.0
        self.drag_start_pos = None
        self.text_info = text
        self.max_zoom = 8.0
        self.min_zoom = 1.1
        self.image_x = 0  # 从缩放图片显示到窗口的偏移x
        self.image_y = 0  # 从缩放图片显示到窗口的偏移y
        self.original_pixmap = None  # 原始图片
        self.scaled_pixmap = None   # 缩放的图片
        self.black_pixmap = None   # 最终显示图片
        self.base_zoom = 1.0  # 基础缩放比例，用于适应窗口大小
        self.image_list = []
        self.current_image_index = -1
        self.current_index_path_old="" # 上次打开的目录
        self.current_index_path = ""
        self.setText(text)

        # 设置初始尺寸
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(900, 800)  # 使用最小尺寸而不是固定尺寸
        self.setMaximumSize(1600, 1600)
        self.initial_size = QSize(self.width(), self.height())
        # 启用鼠标追踪
        self.setMouseTracking(True)
        # 接受键盘信息
        self.setFocusPolicy(Qt.StrongFocus)


    def mouseDoubleClickEvent(self, event: QMouseEvent):
        # 鼠标双击事件
        if event.button() == Qt.LeftButton:
            self.zoom_factor = 1.0
            self.update_image_display()
        if event.button() == Qt.RightButton:
            QMessageBox.information(self, "后处理图片", "当前没有后处理图片")
            print(f"显示处理后的图片")


    def mousePressEvent(self, event: QMouseEvent):
        # 鼠标按下表示可以拖动
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()


    def mouseMoveEvent(self, event: QMouseEvent):
        # 鼠标移动事件
        if event.buttons() & Qt.LeftButton and self.drag_start_pos:
            if self.zoom_factor != 1.0:
                dx = event.x() - self.drag_start_pos.x()
                dy = event.y() - self.drag_start_pos.y()
                self.drag_start_pos = event.pos()
                black_pixmap = move_image(self.scaled_pixmap, self.black_pixmap,
                                          self.image_x, self.image_y, dx, dy)
                # TODO限制拖动区域
                self.image_x = self.image_x + dx
                self.image_y = self.image_y + dy
                self.black_pixmap = black_pixmap
                self.setPixmap(black_pixmap)


    # 鼠标释放事件
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.isDragging = False


    # 滚轮事件
    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            if self.zoom_factor <= self.max_zoom:
                self.zoom_factor += 0.1
        else:
            if self.zoom_factor >= self.min_zoom:
                self.zoom_factor -= 0.1
        self.update_image_display()



    def keyPressEvent(self, event):
        # 键盘事件
        if event.key() == Qt.Key_A:
            self.action_A()
        elif event.key() == Qt.Key_D:
            self.action_D()
        else:
            super().keyPressEvent(event)


    def action_A(self):
        # A 键
        if self.current_image_index >= 0 and self.image_list:
            self.current_image_index = self.current_image_index - 1
            if self.current_image_index < 0:
                self.current_image_index = 0
            self.set_image(self.image_list[self.current_image_index])
        else:
            QMessageBox.information(self, "前一张图片", "索引出现错误")


    def action_D(self):
        # D 键
        if self.current_image_index >= 0 and self.image_list:
            self.current_image_index = self.current_image_index + 1
            if self.current_image_index >= len(self.image_list):
                self.current_image_index = len(self.image_list) - 1
            self.set_image(self.image_list[self.current_image_index])
        else:
            QMessageBox.information(self, "下一张图片", "索引出现错误")


    def set_image(self, image_path):
        try:
            self.original_pixmap = load_image(image_path)
            self.initial_size = QSize(self.width(), self.height())
            self.base_zoom = calculate_base_zoom(
                self.original_pixmap.width(), self.original_pixmap.height(), self.initial_size.width(), self.initial_size.height())
            self.zoom_factor = 1.0
            self.update_image_display()
        except Exception as e:
            ulog().get_logger().info(f"加载图片失败 {str(e)}")


    def update_image_display(self):
        """更新图片显示"""
        if not self.original_pixmap:
            return
        # 如果之前存在，则按照之前的中心进行缩放
        if self.scaled_pixmap:
            w_rate = -self.image_x / self.scaled_pixmap.width()
            h_rate = -self.image_y / self.scaled_pixmap.height()
        # 如果双击事件返回原图，则置零
        if self.zoom_factor == 1.0:
            w_rate = None
            h_rate = None
        self.initial_size = QSize(self.width(), self.height())
        scaled_pixmap, black_pixmap, scaled_width, scaled_height, x, y = scale_image(self.original_pixmap, self.base_zoom, self.zoom_factor,
                                                                                     self.initial_size.width(), self.initial_size.height(), w_rate=w_rate, h_rate=h_rate)
        self.image_x = x
        self.image_y = y
        self.scaled_pixmap = scaled_pixmap
        self.black_pixmap = black_pixmap
        self.pre_scale_pixmap = scaled_pixmap
        self.setPixmap(black_pixmap)



    def receive_siganel_rbc(self, image_path):
        if self.current_index_path_old != self.current_index_path:
            self.current_index_path_old = self.current_index_path
            self.image_list.clear()
            self.current_image_index = 0

        self.current_index_path = image_path
        get_image_list(self.current_index_path + r"\rbc", self.image_list)

        if self.current_image_index > 0:
            self.set_image(self.image_list[self.current_image_index])
        elif self.image_list:
            self.set_image(self.image_list[0])
            self.current_image_index = 0
        else:
            self.set_image("")
            ulog().get_logger().error(
                f"加载图片失败, {self.current_index_path}\rbc 目录下没有图片")

        print(f"当前的图片进度: {self.current_image_index} / {len(self.image_list)} ")


    def receive_siganel_wbc(self, image_path):
        if self.current_index_path_old != self.current_index_path:
            self.current_index_path_old = self.current_index_path
            self.image_list.clear()
            self.current_image_index = 0

        self.current_index_path = image_path
        get_image_list(self.current_index_path + r"\wbc", self.image_list)

        if self.current_image_index > 0:
            self.set_image(self.image_list[self.current_image_index])
        elif self.image_list:
            self.set_image(self.image_list[0])
            self.current_image_index = 0
        else:
            self.set_image("")
            ulog().get_logger().error(
                f"加载图片失败, {self.current_index_path}\wbc 目录下没有图片")

        print(f"当前的图片进度: {self.current_image_index} / {len(self.image_list)} ")


    def receive_siganel_baso(self, image_path):
        if self.current_index_path_old != self.current_index_path:
            self.current_index_path_old = self.current_index_path
            self.image_list.clear()
            self.current_image_index=0

        self.current_index_path = image_path
        get_image_list(self.current_index_path + r"\baso", self.image_list)

        if self.current_image_index > 0:
            self.set_image(self.image_list[self.current_image_index])
        elif self.image_list:
            self.set_image(self.image_list[0])
            self.current_image_index = 0
        else:
            self.set_image("")
            ulog().get_logger().error(
                f"加载图片失败, {self.current_index_path}\baso  目录下没有图片")

        print(f"当前的图片进度: {self.current_image_index} / {len(self.image_list)} ")


    def receive_siganel_ret(self, image_path):
        if self.current_index_path_old != self.current_index_path:
            self.current_index_path_old = self.current_index_path
            self.image_list.clear()

        self.current_image_index = -1
        self.current_index_path = self.current_index_path

        if os.path.exists(self.current_index_path + r"\ret.png"):
            self.set_image(self.current_index_path + r"\ret.png")
        else:
            self.set_image("")
            ulog().get_logger().error(
                f"加载图片失败, {self.current_index_path} 不存在")

        print(f"当前的图片进度: {self.current_image_index} / {len(self.image_list)} ")


    def get_index_process(self):
        if self.image_list and self.current_image_index>=0:
            return self.current_image_index, len(self.image_list), self.image_list[self.current_image_index]
        elif self.text_info == "结果信息":
            return self.current_image_index, len(self.image_list), ""
        else:
            return self.current_image_index, len(self.image_list), "文件索引错误"
