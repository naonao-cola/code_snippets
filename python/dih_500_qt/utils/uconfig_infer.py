import os
import json
from utils.ufile_func import *
from PyQt5.QtWidgets import QMenu, QAction, QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from ui.infer import Ui_Form


class AlgInferEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setup_connections()

    def browse_file_1(self):
        """打开文件对话框并设置路径"""
        from PyQt5.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹",

        )
        if directory:
            self.ui.lineEdit.setText(directory)
            QMessageBox.information(self, "路径", f" 当前选择的路径 {directory}")

    def browse_file_2(self):
        """打开文件对话框并设置路径"""
        from PyQt5.QtWidgets import QFileDialog
        directory = QFileDialog.getExistingDirectory(
            self, "选择图片文件夹",
        )
        if directory:
            self.ui.lineEdit_2.setText(directory)
            QMessageBox.information(self, "路径", f" 当前选择的路径 {directory}")

    def infer(self):
        text_1 = self.ui.lineEdit.text()
        text_2 = self.ui.lineEdit_2.text()

        if text_1 and not text_2:
            QMessageBox.information(self, "推理信息", f" 正在推理 {text_1}")
        if not text_1 and text_2:
            QMessageBox.information(self, "推理信息", f" 正在推理 {text_2}")
        if text_1 and text_2:
            QMessageBox.critical(self, "错误", f"加载配置失败:  只允许推理一个选项")

    def setup_connections(self):
        """连接按钮信号槽"""
        self.ui.pushButton.clicked.connect(self.browse_file_1)
        self.ui.pushButton_2.clicked.connect(self.browse_file_2)
        self.ui.pushButton_3.clicked.connect(self.infer)
