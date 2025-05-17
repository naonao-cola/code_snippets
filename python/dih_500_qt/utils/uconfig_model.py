import os
import json
from utils.ufile_func import *
from PyQt5.QtWidgets import QMenu, QAction, QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from ui.model_cfg import Ui_MODEL_UI


class AlgModConfigEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MODEL_UI()
        self.ui.setupUi(self)
        self.config_path = get_config_path("algo", "alg_model")
        self.config_data = {}

        # 路径输入框列表
        self.path_edits = [
            self.ui.lineEdit_3, self.ui.lineEdit_18, self.ui.lineEdit_22,
            self.ui.lineEdit_26, self.ui.lineEdit_30, self.ui.lineEdit_34,
            self.ui.lineEdit_38, self.ui.lineEdit_42, self.ui.lineEdit_46
        ]

        # 为路径输入框安装事件过滤器
        for edit in self.path_edits:
            edit.installEventFilter(self)

        self.setup_connections()
        self.load_config()

    def eventFilter(self, obj, event):
        """处理路径输入框点击事件"""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QMouseEvent

        # 只处理鼠标按下事件
        if (obj in self.path_edits and event.type() == QEvent.MouseButtonDblClick and isinstance(event, QMouseEvent)):
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
        """从JSON文件加载配置到UI"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)

            # 映射模型ID到对应的UI控件
            model_mapping = {
                0: (self.ui.lineEdit_47, self.ui.lineEdit_45, self.ui.lineEdit_46, self.ui.lineEdit_48),
                1: (self.ui.lineEdit, self.ui.lineEdit_2, self.ui.lineEdit_3, self.ui.lineEdit_4),
                2: (self.ui.lineEdit_27, self.ui.lineEdit_25, self.ui.lineEdit_26, self.ui.lineEdit_28),
                3: (self.ui.lineEdit_23, self.ui.lineEdit_21, self.ui.lineEdit_22, self.ui.lineEdit_24),
                4: (self.ui.lineEdit_39, self.ui.lineEdit_37, self.ui.lineEdit_38, self.ui.lineEdit_40),
                5: (self.ui.lineEdit_43, self.ui.lineEdit_41, self.ui.lineEdit_42, self.ui.lineEdit_44),
                6: (self.ui.lineEdit_19, self.ui.lineEdit_17, self.ui.lineEdit_18, self.ui.lineEdit_20),
                7: (self.ui.lineEdit_35, self.ui.lineEdit_33, self.ui.lineEdit_34, self.ui.lineEdit_36),
                8: (self.ui.lineEdit_31, self.ui.lineEdit_29, self.ui.lineEdit_30, self.ui.lineEdit_32)
            }

            for model in self.config_data.get("ai_models", []):
                model_id = model["model_id"]
                if model_id in model_mapping:
                    id_edit, type_edit, path_edit, conf_edit = model_mapping[model_id]
                    id_edit.setText(str(model["model_id"]))
                    type_edit.setText(str(model["algo_type"]))
                    path_edit.setText(model["model_path"])
                    conf_edit.setText(str(model["conf_threshold"]))
            ## QMessageBox.information(self, "成功", f"读取成功 {self.config_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载配置文件失败: {str(e)}")

    def save_config(self):
        """从UI收集配置并保存到JSON文件"""
        try:
            # 映射模型ID到对应的UI控件
            model_mapping = {
                0: (self.ui.lineEdit_47, self.ui.lineEdit_45, self.ui.lineEdit_46, self.ui.lineEdit_48),
                1: (self.ui.lineEdit, self.ui.lineEdit_2, self.ui.lineEdit_3, self.ui.lineEdit_4),
                2: (self.ui.lineEdit_27, self.ui.lineEdit_25, self.ui.lineEdit_26, self.ui.lineEdit_28),
                3: (self.ui.lineEdit_23, self.ui.lineEdit_21, self.ui.lineEdit_22, self.ui.lineEdit_24),
                4: (self.ui.lineEdit_39, self.ui.lineEdit_37, self.ui.lineEdit_38, self.ui.lineEdit_40),
                5: (self.ui.lineEdit_43, self.ui.lineEdit_41, self.ui.lineEdit_42, self.ui.lineEdit_44),
                6: (self.ui.lineEdit_19, self.ui.lineEdit_17, self.ui.lineEdit_18, self.ui.lineEdit_20),
                7: (self.ui.lineEdit_35, self.ui.lineEdit_33, self.ui.lineEdit_34, self.ui.lineEdit_36),
                8: (self.ui.lineEdit_31, self.ui.lineEdit_29, self.ui.lineEdit_30, self.ui.lineEdit_32)
            }

            for model in self.config_data.get("ai_models", []):
                model_id = model["model_id"]
                if model_id in model_mapping:
                    id_edit, type_edit, path_edit, conf_edit = model_mapping[model_id]
                    model["model_id"] = int(id_edit.text())
                    model["algo_type"] = int(type_edit.text())
                    model["model_path"] = path_edit.text()
                    model["conf_threshold"] = float(conf_edit.text())

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=4)
            QMessageBox.information(self, "成功", "配置保存成功")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置文件失败: {str(e)}")

    def setup_connections(self):
        """连接按钮信号槽"""
        self.ui.pushButton.clicked.connect(self.load_config)
        self.ui.pushButton_2.clicked.connect(self.save_config)
