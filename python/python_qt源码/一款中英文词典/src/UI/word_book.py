from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QComboBox, QPushButton, QCheckBox, QTableWidget, QInputDialog, QAbstractItemView, \
    QHeaderView, QTableWidgetItem, QDialog

from src.UI.review import UiReview
from src.UI.util import create_line, create_multi_line
from src.backend.word_book import WordBook
from src.events import events


class UiSelectGroup(QDialog):
    group_id = -1

    def __init__(self, parent, word_book):
        super().__init__(parent)
        self.combo_groups = QComboBox()
        self.combo_groups.clear()
        for group in word_book.get_groups():
            self.combo_groups.addItem(group[1], group[0])

        self.btn_ok = QPushButton("  确定  ")
        self.btn_ok.clicked.connect(self.on_ok)
        self.btn_cancel = QPushButton("  取消  ")
        self.btn_cancel.clicked.connect(self.reject)

        layout = create_multi_line([
            create_line(["分类", self.combo_groups]),
            create_line([1, self.btn_ok, self.btn_cancel, 1]),
        ])
        self.setLayout(layout)

    def on_ok(self):
        self.group_id = self.combo_groups.currentData()
        self.accept()


class UiWordBook(QWidget):
    groups = {}

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.word_book = WordBook()
        events.signal_add_2_wordbook.connect(self.on_add_word)

        self.combo_groups = QComboBox()
        self.combo_groups.currentIndexChanged.connect(self.init_words)

        self.button_review = QPushButton("  复习生词  ")
        self.button_review.clicked.connect(self.on_review)

        self.checkbox_select_all = QCheckBox("全选")
        self.checkbox_select_all.clicked.connect(self.on_select_all)
        self.button_add_group = QPushButton("  添加分类  ")
        self.button_add_group.clicked.connect(self.on_add_group)
        self.button_delete_group = QPushButton("  删除分类  ")
        self.button_delete_group.clicked.connect(self.on_delete_group)

        self.button_delete_word = QPushButton("  删除生词  ")
        self.button_delete_word.clicked.connect(self.on_delete_word)
        self.button_change_group = QPushButton("  更改分类  ")
        self.button_change_group.clicked.connect(self.on_change_group)

        self.table = QTableWidget()
        header = ["单词", "翻译", "添加时间", "复习次数", "分类"]
        self.table.setColumnCount(len(header))
        self.table.setHorizontalHeaderLabels(header)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setResizeContentsPrecision(QHeaderView.ResizeToContents)
        self.table.itemDoubleClicked.connect(self.on_double_clicked)

        tools = create_line([
            self.checkbox_select_all,
            self.button_delete_word,
            self.button_change_group,
            self.button_review,
            1,
            self.combo_groups,
            self.button_add_group,
            self.button_delete_group,
        ])

        self.setLayout(
            create_multi_line([
                tools,
                self.table
            ])
        )
        self.init_groups()

    def init_groups(self):
        self.combo_groups.clear()
        self.groups.clear()
        groups = self.word_book.get_groups()
        for group in groups:
            self.groups[group[0]] = group[1]
        self.combo_groups.addItem("所有分类", -1)
        for group in groups:
            self.combo_groups.addItem(group[1], group[0])

        self.update_ui()
        self.init_words()
        return True

    def update_ui(self):
        self.button_delete_group.setEnabled(self.combo_groups.currentData() > 1)

    def init_words(self):
        if self.combo_groups.currentIndex() < 0:
            return
        for i in range(self.table.rowCount()):
            self.table.removeRow(0)
        for word in self.word_book.get_words(self.combo_groups.currentData()):
            self.table.insertRow(0)
            word_id = word[0]
            group_id = self.word_book.get_group(word_id)
            if len(group_id) > 0:
                group_id = group_id[0][0]
            else:
                group_id = 1
            for i in range(4):
                item = QTableWidgetItem(str(word[i + 1]))
                if i == 0:
                    item.setCheckState(Qt.Unchecked)
                    item.setData(Qt.UserRole, word_id)
                    item.setData(Qt.UserRole + 1, group_id)
                elif i == 1:
                    item.setToolTip(str(word[i + 1]))
                self.table.setItem(0, i, item)
            self.table.setItem(0, 4, QTableWidgetItem(str(self.groups[group_id])))
        self.table.resizeColumnsToContents()
        self.update_ui()
        return True

    def on_select_all(self, checked):
        checked = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(checked)

    def on_add_group(self):
        name, succeed = QInputDialog.getText(self, "新增分类", "分类名称")
        if name is None or not succeed:
            return
        self.word_book.create_group(name)
        self.init_groups()

    def on_delete_group(self):
        if self.combo_groups.currentIndex() < 0:
            return
        self.word_book.delete_group(self.combo_groups.currentData())
        self.init_groups()

    def on_add_word(self, word, translate):
        word_id = self.word_book.add_word(word, translate)
        self.word_book.add_word_to_group(word_id, 1)
        self.init_groups()

    def on_change_group(self):
        d = UiSelectGroup(self, self.word_book)
        if d.exec_() != 1:
            return

        for i in range(self.table.rowCount() - 1, -1, -1):
            item = self.table.item(i, 0)
            if item.checkState() != Qt.Checked:
                continue
            word_id = item.data(Qt.UserRole)
            group_id = item.data(Qt.UserRole + 1)
            self.word_book.add_word_to_group(word_id, d.group_id)
            self.word_book.delete_word_from_group(word_id, group_id)

        self.init_words()

    def on_delete_word(self):
        # 注意：只能从后往前删除
        for i in range(self.table.rowCount() - 1, -1, -1):
            item = self.table.item(i, 0)
            if item.checkState() != Qt.Checked:
                continue
            self.word_book.delete_word(item.data(Qt.UserRole))
            self.table.removeRow(i)

    def on_double_clicked(self, item: QTableWidgetItem):
        item = self.table.item(item.row(), 0)
        item.setCheckState(Qt.Checked if item.checkState() != Qt.Checked else Qt.Unchecked)

    def on_review(self):
        d = UiReview(self, self.combo_groups.currentData(), self.word_book)
        d.exec_()
        self.init_words()
