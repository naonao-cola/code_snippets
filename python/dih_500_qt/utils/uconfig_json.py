import os
import json
from utils.ufile_func import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QComboBox, QAction, QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QMessageBox

# 算法配置文件
# 对应于 config/algo/alg_cfg 文件夹下的 JSON 文件


condition_symbol = [">=", "<="]
condition_enable = ["True", "False"]
condition_name = [
    "wbc_num",
    "neu_num",
    "lym_num",
    "mon_num",
    "eos_num",
    "baso_num",
    "nrbc_num",
    "ig_num",
    "rbc_rdw_sd",
    "rbc_rdw_cv",
    "rbc_mcv",
    "rbc_num",
    "rbc_hgb",
    "rbc_mchc",
    "plt_num",
    "ret_num",
]


class JsonConfigEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("检测条件")
        self.resize(440, 400)

        layout = QVBoxLayout()
        # json 的 读取等
        self.config_path = get_config_path("algo", "alg_json")
        # 获取检测条件

        # 初始化UI
        self.tableWidget = QTableWidget()
        self.init_ui()
        layout.addWidget(self.tableWidget)

        # 水平
        horizontal_layout = QHBoxLayout()
        self.pushButton_1 = QtWidgets.QPushButton()
        self.pushButton_1.setGeometry(QtCore.QRect(450, 1270, 101, 23))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_1.setText("读取检测条件")
        self.pushButton_2 = QtWidgets.QPushButton()
        self.pushButton_2.setGeometry(QtCore.QRect(450, 1270, 101, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("应用检测条件")
        horizontal_layout.addWidget(self.pushButton_1)
        horizontal_layout.addWidget(self.pushButton_2)
        layout.addLayout(horizontal_layout)
        self.setLayout(layout)
        # 设置大小
        self.setMinimumSize(QtCore.QSize(440, 400))
        self.setMaximumSize(QtCore.QSize(440, 400))
        # 连接按钮信号
        self.pushButton_1.clicked.connect(self.set_condition_combox_list)
        self.pushButton_2.clicked.connect(self.set_condition_json_list)

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.json_content = json.load(f)
            self.json_condition_content = self.json_content["judgements"]
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")

    def init_ui(self, col_num=4, row_num=32):
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.tableWidget.setFont(font)
        self.tableWidget.setColumnCount(col_num)
        self.tableWidget.setRowCount(row_num)
        self.tableWidget.setHorizontalHeaderLabels(
            ['name', 'enable', 'symbol', 'value'])
        self.condition_combox_list = []
        for i in range(row_num):
            combox_01, combox_02, combox_03 = self.generate_combox()
            # 设置默认选项
            combox_01.setCurrentIndex(i/2)
            combox_02.setCurrentIndex(1)
            if i / 2 == 0:
                combox_03.setCurrentIndex(0)
            else:
                combox_03.setCurrentIndex(1)
            # 复选框
            self.tableWidget.setCellWidget(i, 0, combox_01)
            self.tableWidget.setCellWidget(i, 1, combox_02)
            self.tableWidget.setCellWidget(i, 2, combox_03)
            # 编辑框
            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(i, 3, item)
            self.condition_combox_list.append({
                "combox_01": combox_01,
                "combox_02": combox_02,
                "combox_03": combox_03,
                "item": item
            })

        self.load_config()
        self.set_condition_combox_list()

    def set_condition_combox_list(self):
        # condition_combox_count = len(self.condition_combox_list)
        self.load_config()
        json_used_index = [0] * \
            len(self.json_condition_content[0]["judgeparams"])
        for m, combox_item in enumerate(self.condition_combox_list):
            # 是否存在的标志位 #   ['name', 'enable', 'symbol', 'value'])
            exit_flag = 0
            combox_01_txt = combox_item["combox_01"].currentText()
            combox_02_txt = combox_item["combox_02"].currentText()
            combox_03_txt = combox_item["combox_03"].currentText()
            item = combox_item["item"].text()
            for n, json_item in enumerate(self.json_condition_content[0]["judgeparams"]):
                # 启用
                if combox_01_txt == json_item["name"] and json_item["enable"] and not json_used_index[n]:
                    combox_item["combox_02"].setCurrentIndex(0)
                    combox_item["combox_03"].setCurrentIndex(
                        condition_symbol.index(json_item["symbol"]))
                    combox_item["item"].setText(str(json_item["value"]))
                    # print(f"当前的序号 {n}  {json_item} ")
                    exit_flag = 1
                    json_used_index[n] = 1
                    break
            if not exit_flag:
                combox_item["combox_02"].setCurrentIndex(1)
                combox_item["item"].setText("0")
        # QMessageBox.information(self, "成功", f"读取成功： {self.config_path}")

    def set_condition_json_list(self):
        self.load_config()
        write_data = []
        for m, combox_item in enumerate(self.condition_combox_list):
            combox_01_txt = combox_item["combox_01"].currentText()
            combox_02_txt = combox_item["combox_02"].currentText()
            combox_03_txt = combox_item["combox_03"].currentText()
            item = combox_item["item"].text()
            if combox_02_txt == "True":  # 如果启用再写入文件
                write_data.append({
                    "name": combox_01_txt,
                    "enable": combox_02_txt,
                    "symbol": combox_03_txt,
                    "value": item
                })
        self.json_condition_content[0]["judgeparams"] = write_data
        self.json_content["judgements"] = self.json_condition_content
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_content, f, indent=4)

        # QMessageBox.information(self, "成功", f"设置成功： {self.config_path}")

    def generate_combox(self):
        combox_01 = QComboBox()
        for idx, item in enumerate(condition_name):
            combox_01.addItem(item)

        combox_02 = QComboBox()
        for idx, item in enumerate(condition_enable):
            combox_02.addItem(item)

        combox_03 = QComboBox()
        for idx, item in enumerate(condition_symbol):
            combox_03.addItem(item)

        combox_01.setEditable(True)  # 居中设置
        combox_02.setEditable(True)
        combox_03.setEditable(True)
        combox_01.lineEdit().setReadOnly(True)
        combox_02.lineEdit().setReadOnly(True)
        combox_03.lineEdit().setReadOnly(True)
        combox_01.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        combox_02.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        combox_03.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        return combox_01, combox_02, combox_03
