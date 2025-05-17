import os
import json
from utils.ufile_func import *
from PyQt5.QtWidgets import QMenu, QAction, QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from ui.model_cfg_2 import Ui_Form


class AlgModConfigEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # cbc
        self.path_edit_cbc = [
            self.ui.lineEdit_55, self.ui.lineEdit_57, self.ui.lineEdit_59,
            self.ui.lineEdit_61, self.ui.lineEdit_63, self.ui.lineEdit_65,
            self.ui.lineEdit_67, self.ui.lineEdit_69, self.ui.lineEdit_71
        ]
        self.path_param_cbc = [
            self.ui.lineEdit_56, self.ui.lineEdit_58, self.ui.lineEdit_60,
            self.ui.lineEdit_62, self.ui.lineEdit_64, self.ui.lineEdit_66,
            self.ui.lineEdit_68, self.ui.lineEdit_70, self.ui.lineEdit_72
        ]
        self.putton_cbc = [
            self.ui.pushButton_7, self.ui.pushButton_8
        ]

        # cbcp
        self.path_edit_cbcp = [
            self.ui.lineEdit_37, self.ui.lineEdit_39, self.ui.lineEdit_41,
            self.ui.lineEdit_43, self.ui.lineEdit_45, self.ui.lineEdit_47,
            self.ui.lineEdit_49, self.ui.lineEdit_51, self.ui.lineEdit_53
        ]
        self.path_param_cbcp = [
            self.ui.lineEdit_38, self.ui.lineEdit_40, self.ui.lineEdit_42,
            self.ui.lineEdit_44, self.ui.lineEdit_46, self.ui.lineEdit_48,
            self.ui.lineEdit_50, self.ui.lineEdit_52, self.ui.lineEdit_54
        ]

        self.putton_cbcp = [
            self.ui.pushButton_5, self.ui.pushButton_6
        ]

        # cbcr
        self.path_edit_cbcr = [
            self.ui.lineEdit_19, self.ui.lineEdit_21, self.ui.lineEdit_23,
            self.ui.lineEdit_25, self.ui.lineEdit_27, self.ui.lineEdit_29,
            self.ui.lineEdit_31, self.ui.lineEdit_33, self.ui.lineEdit_35
        ]
        self.path_param_cbcr = [
            self.ui.lineEdit_20, self.ui.lineEdit_22, self.ui.lineEdit_24,
            self.ui.lineEdit_26, self.ui.lineEdit_28, self.ui.lineEdit_30,
            self.ui.lineEdit_32, self.ui.lineEdit_34, self.ui.lineEdit_36
        ]
        self.putton_cbcr = [
            self.ui.pushButton_4, self.ui.pushButton_5
        ]

         # common
        self.path_edit_common = [
            self.ui.lineEdit, self.ui.lineEdit_3, self.ui.lineEdit_5,
            self.ui.lineEdit_7, self.ui.lineEdit_9, self.ui.lineEdit_11,
            self.ui.lineEdit_13, self.ui.lineEdit_15, self.ui.lineEdit_17
        ]

        self.path_param_common = [
            self.ui.lineEdit_2, self.ui.lineEdit_4, self.ui.lineEdit_6,
            self.ui.lineEdit_8, self.ui.lineEdit_10, self.ui.lineEdit_12,
            self.ui.lineEdit_14, self.ui.lineEdit_16, self.ui.lineEdit_18
        ]
        self.putton_common = [
            self.ui.pushButton, self.ui.pushButton_2
        ]

        ## 资源列表
        self.path_edit=[]
        self.path_edit.append(self.path_edit_cbc)
        self.path_edit.append(self.path_edit_cbcp)
        self.path_edit.append(self.path_edit_cbcr)
        self.path_edit.append(self.path_edit_common)

        self.path_param = []
        self.path_param.append(self.path_param_cbc)
        self.path_param.append(self.path_param_cbcp)
        self.path_param.append(self.path_param_cbcr)
        self.path_param.append(self.path_param_common)

        self.putton = []
        self.putton.append(self.putton_cbc)
        self.putton.append(self.putton_cbcp)
        self.putton.append(self.putton_cbcr)
        self.putton.append(self.putton_common)


        # 为路径输入框安装事件过滤器
        for vec_item in self.path_edit:
            for edit in vec_item:
                edit.installEventFilter(self)

        self.setup_connections()
        self.load_config()



    def eventFilter(self, obj, event):
        """处理路径输入框点击事件"""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QMouseEvent

        # 只处理鼠标按下事件
        exit_flag=False
        for vec_item in self.path_edit:
            if obj in vec_item:
                exit_flag= True

        if (exit_flag and event.type() == QEvent.MouseButtonDblClick and isinstance(event, QMouseEvent)):
            # 确保是左键点击
            if event.button() == Qt.LeftButton:
                self.browse_file(obj)
                return True
        # 其他事件交给父类处理
        return super(AlgModConfigEditor, self).eventFilter(obj, event)



    def browse_file(self, line_edit):
        """打开文件对话框并设置路径"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件",
            os.path.dirname(line_edit.text()) if line_edit.text() else "",
            "Engine Files (*.engine);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)



    def load_config(self):
        """从JSON文件加载配置到UI
        """
        ## 当前选定的是哪一个
        label_index = self.ui.tabWidget.currentIndex()
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)




        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置文件失败: {str(e)}")

    def save_config(self):
        """从UI收集配置并保存到JSON文件
        """
        label_index = self.ui.tabWidget.currentIndex()
        try:
            QMessageBox.information(self, "成功", "配置保存成功")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置文件失败: {str(e)}")

    def setup_connections(self):
        """连接按钮信号槽"""
        self.ui.pushButton.clicked.connect(self.load_config)
        self.ui.pushButton_3.clicked.connect(self.load_config)
        self.ui.pushButton_5.clicked.connect(self.load_config)
        self.ui.pushButton_5.clicked.connect(self.load_config)

        self.ui.pushButton_2.clicked.connect(self.save_config)
        self.ui.pushButton_4.clicked.connect(self.save_config)
        self.ui.pushButton_6.clicked.connect(self.save_config)
        self.ui.pushButton_8.clicked.connect(self.save_config)
