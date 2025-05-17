import os
import json
from utils.ufile_func import *
from PyQt5.QtWidgets import QMenu, QAction, QDialog, QFileDialog, QTextEdit, QPushButton, QMessageBox, QAbstractItemView, QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal
from ui.sample_file import Ui_widget
from utils.database import DataEdit

# 左侧样本列表


class SampleEditor(QDialog):

    folder_selected = pyqtSignal(str)  # 添加信号

    def __init__(self, db: DataEdit, parent=None, ):
        super().__init__(parent)
        self.ui = Ui_widget()
        self.ui.setupUi(self)
        # UI 设置
        # 设置选择整行
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)

        self.ui.tableWidget.itemDoubleClicked.connect(self.get_date)
        self.db = db
        self.start_idx = -1
        self.end_idx = -1

        # 文件
        self.open_directories = []
        # 连接按钮信号
        self.ui.pushButton.clicked.connect(self.previous_page)
        self.ui.pushButton_2.clicked.connect(self.next_page)
        self.ui.pushButton_3.clicked.connect(self.detail_page)

    def open_directory(self):
        # 打开文件夹对话框
        directory = QFileDialog.getExistingDirectory(self, "选择 SN 图片文件夹")
        if directory:
            self.scan(directory)
            self.write_item()

    def add_directory(self):
        # 打开文件夹对话框
        directory = QFileDialog.getExistingDirectory(self, "选择 SN 图片文件夹")
        if directory:
            self.scan(directory)
            self.write_item()

    def delete_directory(self):
        """只支持删除sn目录
        """
        ##TODO 对路径进行校验

        text, ok = QInputDialog.getText(self, '删除目录', '请输入删除的目录(只支持删除sn目录):')

        if ok:
            QMessageBox.information(self, 'Result', f"删除的目录: {text}")
            self.db.auto_del(text)
        else:
            QMessageBox.information(self, 'Result', "You canceled the dialog.")



    def scan(self, directory):
        self.db.db_connect()
        self.db.add_path(directory)
        self.db.auto_add()
        # 字典列表
        self.open_directories = self.db.auto_sea_all()

    def write_item(self):
        """填充表格
        """
        column_count = self.ui.tableWidget.columnCount()
        row_count = self.ui.tableWidget.rowCount()
        self.start_idx = 0
        if self.open_directories:
            data_idx = 0
            for r in range (0,row_count,1):
                if data_idx < len(self.open_directories):
                    data_item = self.open_directories[data_idx]
                    sn_str = data_item.get("sn")
                    date_str = data_item.get("date")
                    sample_str = data_item.get("sample_id")
                    categorize_str = data_item.get("categorize")
                    # self.ui.tableWidget.item(r, d_col).text()
                    self.ui.tableWidget.item(r, 0).setText(sample_str)
                    self.ui.tableWidget.item(r, 1).setText(date_str)
                    self.ui.tableWidget.item(r, 2).setText(categorize_str)

                    data_idx += 1
            print(f"已经填充的行数 {data_idx} 行")
            self.end_idx = data_idx
            # ? 其余行数填充无测试
            # TODO 后续注释掉
            for r in range(data_idx, row_count, 1):
                self.ui.tableWidget.item(r, 0).setText("null")
                self.ui.tableWidget.item(r, 1).setText("null")
                self.ui.tableWidget.item(r, 2).setText("null")
            pass
        else:
            print(f"填充表格出错")
        pass

    def write_range_item(self, start, end=None):
        """从指定的序号填充列表
        """
        column_count = self.ui.tableWidget.columnCount()
        row_count = self.ui.tableWidget.rowCount()
        self.start_idx = start
        if self.open_directories:
            data_idx = start
            for r in row_count:
                if data_idx < len(self.open_directories):
                    data_item = self.open_directories[data_idx]
                    sn_str = data_item.get("sn")
                    date_str = data_item.get("date")
                    sample_str = data_item.get("sample_id")
                    categorize_str = data_item.get("categorize")
                    self.ui.tableWidget.item(r, 0).setText(sample_str)
                    self.ui.tableWidget.item(r, 1).setText(date_str)
                    self.ui.tableWidget.item(r, 2).setText(categorize_str)
                    data_idx += 1
            print(f"已经填充的行数 {data_idx - self.start_idx} 行")
            self.end_idx = data_idx
            # ? 其余行数填充无测试
            # TODO 后续注释掉
            # 剩余多少行
            remaining = self.end_idx - self.start_idx
            for r in range(remaining, row_count, 1):
                self.ui.tableWidget.item(r, 0).setText("null")
                self.ui.tableWidget.item(r, 1).setText("null")
                self.ui.tableWidget.item(r, 2).setText("null")
            pass
        else:
            print(f"填充表格出错")
        pass

    def previous_page(self):
        """上一页
        """
        if self.start_idx <= self.ui.tableWidget.rowCount():
            print(f"当前的开始序号：{self.start_idx} 结束序号： {self.end_idx} ")
            QMessageBox.warning(
                self, "错误", f"翻页错误: 当前的开始序号：{self.start_idx} 结束序号： {self.end_idx}")
            return
        else:
            # 翻页一整页
            self.start_idx -= self.ui.tableWidget.rowCount()
            self.write_range_item(self.start_idx)

    def next_page(self):
        """ 下一页
        """
        if self.end_idx + self.ui.tableWidget.rowCount() >= len(self.open_directories):
            print(f"当前的开始序号：{self.start_idx} 结束序号： {self.end_idx} ")
            QMessageBox.warning(
                self, "错误", f"翻页错误: 当前的开始序号：{self.start_idx} 结束序号： {self.end_idx}")
            return
        else:
            # 翻页一整页
            self.start_idx += self.ui.tableWidget.rowCount()
            self.write_range_item(self.start_idx)

    def detail_page(self):
        """ 详情页

        """
        pass

    def get_date(self, Item=None):
        if Item is None:
            return
        else:
            row = Item.row()  # 获取行数
            col = Item.column()  # 获取列数 注意是column而不是col哦
            text = Item.text()  # 获取内容
            print(f"当前选中的行: {row},当前选中的列： {col} , 内容： {text}")
            sample_id = self.ui.tableWidget.item(row, 0).text()
            date_str = self.ui.tableWidget.item(row, 1).text()
            cate_str = self.ui.tableWidget.item(row, 2).text()
            result = self.db.auto_sea(date= date_str, sample_id=sample_id,categorize=cate_str)
            if result:
                path = result[0].get("path")
                print(f"当前选中的路径{path}")
                self.folder_selected.emit(path)
            else:
                print(f"当前选中的路径为空 {result}")
