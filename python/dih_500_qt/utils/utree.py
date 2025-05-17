#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from PyQt5.QtWidgets import QTreeWidget, QFileDialog, QTreeWidgetItem, QDirModel, QTreeView
from PyQt5.QtCore import Qt, pyqtSignal
from utils import ufile_func


# 目录树标签
class UTree(QTreeWidget):
    folder_selected = pyqtSignal(str)  # 添加信号

    def __init__(self, parent=None):
        super().__init__(parent)
        # 当前打开的目录列表
        self.open_directories = []
        # 当前图片路径和JSON路径
        self.current_image_path = None
        self.current_json_path = None
        # 当前图片所属的最顶层目录
        self.current_top_directory = None
        # 第一个标签显示的图片的所属目录
        self.first_label_image_directory = None
        self.setHeaderLabels(["文件列表"])
        self.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.setStyleSheet(
            """
            QTreeWidget {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
            }
            QTreeWidget::item {
                padding: 5px;
                margin: 2px;
                color: #333333;
            }
            QTreeWidget::item:selected {
                background-color: #a0c4ff;
                color: #ffffff;
                border: 1px solid #808080;
            }
            QTreeWidget::branch:selected {
                background-color: #a0c4ff;
            }
            QTreeWidget::branch:hover {
                background-color: #e6f2ff;
            }
        """
        )

    def get_top_directory(self, file_path):
        # 获取文件所属的最顶层目录
        item = self.currentItem()
        while item.parent():
            item = item.parent()
        return item.data(0, Qt.UserRole)

    def open_directory(self):
        # 清空目录树
        self.clear()
        self.open_directories = []
        # 打开文件夹对话框
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if directory:
            self.add_directory_to_tree(directory)

    def add_directory(self):
        # 打开文件夹对话框
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if directory:
            self.add_directory_to_tree(directory)

    def add_directory_to_tree(self, directory):
        # 检查目录是否已经打开
        if directory in self.open_directories:
            return
        # 添加到已打开的目录列表
        self.open_directories.append(directory)
        # 获取目录内容
        root_item = QTreeWidgetItem(self)
        root_item.setText(0, os.path.basename(directory))
        root_item.setData(0, Qt.UserRole, directory)
        # 递归加载目录内容
        self.load_directory_recursive(directory, root_item)
        # 展开根节点
        root_item.setExpanded(True)

    def load_directory_recursive(self, directory, parent_item):
        # 获取目录中的文件和文件夹
        entries = os.listdir(directory)
        entries.sort()

        for entry in entries:
            path = os.path.join(directory, entry)
            if os.path.isdir(path):
                # 如果是文件夹，递归加载
                folder_item = QTreeWidgetItem(parent_item)
                folder_item.setText(0, entry)
                folder_item.setData(0, Qt.UserRole, path)
                folder_item.setExpanded(False)

                # 检查文件夹中是否有图片
                has_images = any(ufile_func.is_image_file(os.path.join(path, f)) for f in os.listdir(
                    path) if os.path.isfile(os.path.join(path, f)))
                if not has_images:
                    folder_item.setForeground(0, Qt.gray)  # 空目录显示灰色
                    folder_item.setText(0, f"{entry}")

                # 递归加载子文件夹
                self.load_directory_recursive(path, folder_item)
            elif os.path.isfile(path) and ufile_func.is_image_file(path):
                # 如果是图片文件，添加到目录树
                file_item = QTreeWidgetItem(parent_item)
                file_item.setText(0, entry)
                file_item.setData(0, Qt.UserRole, path)

    def on_tree_item_double_clicked(self, item, column):
        # 双击目录树项时的处理
        file_path = item.data(0, Qt.UserRole)
        if os.path.isfile(file_path):
            self.current_image_path = file_path
            self.folder_selected.emit(file_path)
            # 更新当前图片所属的最顶层目录
            self.current_top_directory = self.get_top_directory(file_path)

            # 如果这是第一个加载的图片，更新第一个标签显示的图片的所属目录
            if self.first_label_image_directory is None:
                self.first_label_image_directory = os.path.dirname(file_path)

    def get_top_directory(self, file_path):
        # 获取文件所属的最顶层目录
        item = self.currentItem()
        while item.parent():
            item = item.parent()
        return item.data(0, Qt.UserRole)

    def close_directory(self):
        # 获取当前选中的项
        selected_item = self.currentItem()
        if not selected_item:
            return
        # 获取选中的根节点
        root_item = selected_item
        while root_item.parent():
            root_item = root_item.parent()
        # 获取目录路径
        directory = root_item.data(0, Qt.UserRole)
        # 从目录列表中移除
        if directory in self.open_directories:
            self.open_directories.remove(directory)
        # 从目录树中移除
        index = self.indexOfTopLevelItem(root_item)
        if index >= 0:
            self.takeTopLevelItem(index)
