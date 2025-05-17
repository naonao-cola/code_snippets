#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem, QLabel,
    QVBoxLayout, QWidget, QFileDialog, QMenu, QAction, QSplitter, QTabWidget, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QMouseEvent, QWheelEvent
from PyQt5.QtCore import Qt, QPoint, QTimer, QDateTime
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image

from utils import ulog, uimage, ustyle, usample_file, uconfig_alg, uconfig_model, uconfig_json, uconfig_infer, database


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = ulog.ulog().get_logger()
        self.setWindowTitle("DIH-500")
        self.setGeometry(100, 100, 1200, 850)
        self.setMinimumSize(QtCore.QSize(500, 850))
        self.setMaximumSize(QtCore.QSize(6625, 850))
        self.db = database.DataEdit()
        # 创建主窗口的布局
        self.splitter = QSplitter(Qt.Horizontal)
        # 左侧的目录树
        # self.tree_widget = utree.UTree()
        self.tree_widget = usample_file.SampleEditor(self.db)

        # 中间的标签页
        self.tab_widget = QTabWidget()
        # 图片显示标签页
        self.image_label_1 = uimage.UImage("RBC")
        self.image_label_2 = uimage.UImage("WBC")
        self.image_label_3 = uimage.UImage("BASO")
        # 添加标签页到标签页小部件
        self.tab_widget.addTab(self.image_label_1, "RBC")
        self.tab_widget.addTab(self.image_label_2, "WBC")
        self.tab_widget.addTab(self.image_label_3, "BASO")


        # 右侧标签
        self.info_label = QLabel("百分比信息")
        self.info_label.setMinimumSize(QtCore.QSize(200, 850))
        self.info_label.setMaximumSize(QtCore.QSize(300, 850))
        # 添加到分割器
        self.splitter.addWidget(self.tree_widget)
        self.splitter.addWidget(self.info_label)

        self.splitter.addWidget(self.tab_widget)
        self.splitter.setSizes([300, 100, 800])

        # 设置主窗口的中心部件
        self.setCentralWidget(self.splitter)
        # 创建菜单栏
        self.create_menu()
        self.create_status()
        self.setStyleSheet(ustyle.apply_windows_style())

        # 事件相关联
        self.tree_widget.folder_selected.connect(
            self.image_label_1.receive_siganel_rbc)
        self.tree_widget.folder_selected.connect(
            self.image_label_2.receive_siganel_wbc)
        self.tree_widget.folder_selected.connect(
            self.image_label_3.receive_siganel_baso)


    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("文件")
        open_dir_action = QAction("打开目录", self)
        open_dir_action.triggered.connect(self.tree_widget.open_directory)
        file_menu.addAction(open_dir_action)
        add_dir_action = QAction("添加目录", self)
        add_dir_action.triggered.connect(self.tree_widget.add_directory)
        file_menu.addAction(add_dir_action)
        close_dir_action = QAction("删除目录", self)
        close_dir_action.triggered.connect(self.tree_widget.delete_directory)
        file_menu.addAction(close_dir_action)

        # 算法配置
        algo_menu = menu_bar.addMenu("配置")
        alg_config_action = QAction("算法配置", self)
        alg_config_action.triggered.connect(self.show_alg_config_dialog)
        algo_menu.addAction(alg_config_action)
        # 模型配置
        alg_cfg_model_action = QAction("模型配置", self)
        alg_cfg_model_action.triggered.connect(self.show_model_config_dialog)
        algo_menu.addAction(alg_cfg_model_action)
        # 检测条件配置
        alg_json_condition_action = QAction("报警条件", self)
        alg_json_condition_action.triggered.connect(
            self.show_model_json_dialog)
        algo_menu.addAction(alg_json_condition_action)

        # 推理设置
        algo_infer_menu = menu_bar.addMenu("推理")
        alg_infer_action = QAction("推理设置", self)
        alg_infer_action.triggered.connect(self.show_infer_dialog)
        algo_infer_menu.addAction(alg_infer_action)

    def create_status(self):
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("准备就绪")
        self.status_bar_timer = QTimer(self)
        self.status_bar_timer.timeout.connect(self.updateStatusBar)
        self.status_bar_timer.start(500)  # 每隔0.5秒更新状态栏信息

    def updateStatusBar(self):
        label_index = self.tab_widget.currentIndex()
        idx = 0
        img_count = 0
        img_pth = ""
        if label_index == 0:
            idx, img_count, img_pth = self.image_label_1.get_index_process()
        if label_index == 1:
            idx, img_count, img_pth = self.image_label_2.get_index_process()
        if label_index == 2:
            idx, img_count, img_pth = self.image_label_3.get_index_process()
        current_time = QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')
        self.status_bar.showMessage(
            f'当前时间: {current_time}   当前索引：{idx} / {img_count}  当前文件： {img_pth}')

    def show_alg_config_dialog(self):
        dialog = uconfig_alg.AlgConfigEditor(self)
        dialog.exec_()

    def show_model_config_dialog(self):
        dialog = uconfig_model.AlgModConfigEditor(self)
        dialog.exec_()

    def show_model_json_dialog(self):
        dialog = uconfig_json.JsonConfigEditor(self)
        dialog.exec_()

    def show_infer_dialog(self):
        dialog = uconfig_infer.AlgInferEditor(self)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
