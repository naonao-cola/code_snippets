# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Chat.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from module.items import *

class Ui_ChatForm(object):
    def setupUi(self, ChatForm):
        ChatForm.setObjectName("ChatForm")
        ChatForm.resize(1400, 900)
        ChatForm.setStyleSheet("")
        self.send_Button = QtWidgets.QPushButton(ChatForm)
        self.send_Button.setGeometry(QtCore.QRect(1010, 840, 75, 41))
        self.send_Button.setText("")
        self.send_Button.setObjectName("send_Button")
        self.textBrowser = QtWidgets.QTextBrowser(ChatForm)
        self.textBrowser.setGeometry(QtCore.QRect(430, 90, 681, 631))
        self.textBrowser.setObjectName("textBrowser")
        self.textEdit = QtWidgets.QTextEdit(ChatForm)
        self.textEdit.setGeometry(QtCore.QRect(430, 740, 681, 101))
        self.textEdit.setObjectName("textEdit")
        self.pushButton1 = QtWidgets.QPushButton(ChatForm)
        self.pushButton1.setGeometry(QtCore.QRect(150, 210, 75, 23))
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton1_2 = QtWidgets.QPushButton(ChatForm)
        self.pushButton1_2.setGeometry(QtCore.QRect(150, 170, 75, 23))
        self.pushButton1_2.setObjectName("pushButton1_2")
        self.f_avatar_label = QtWidgets.QLabel(ChatForm)
        self.f_avatar_label.setGeometry(QtCore.QRect(1170, 40, 171, 171))
        self.f_avatar_label.setStyleSheet("")
        self.f_avatar_label.setText("")
        self.f_avatar_label.setScaledContents(True)
        self.f_avatar_label.setObjectName("f_avatar_label")
        self.f_nickname_label = QtWidgets.QLabel(ChatForm)
        self.f_nickname_label.setGeometry(QtCore.QRect(1170, 220, 181, 41))
        self.f_nickname_label.setText("")
        self.f_nickname_label.setObjectName("f_nickname_label")
        self.chatname_label = QtWidgets.QLabel(ChatForm)
        self.chatname_label.setGeometry(QtCore.QRect(450, 20, 191, 41))
        self.chatname_label.setText("")
        self.chatname_label.setObjectName("chatname_label")
        self.verticalLayoutWidget = QtWidgets.QWidget(ChatForm)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(1200, 350, 160, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.address_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.address_label.setText("")
        self.address_label.setObjectName("address_label")
        self.verticalLayout.addWidget(self.address_label)
        self.email_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.email_label.setText("")
        self.email_label.setObjectName("email_label")
        self.verticalLayout.addWidget(self.email_label)
        self.tel_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.tel_label.setText("")
        self.tel_label.setObjectName("tel_label")
        self.verticalLayout.addWidget(self.tel_label)
        self.birthday_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.birthday_label.setText("")
        self.birthday_label.setObjectName("birthday_label")
        self.verticalLayout.addWidget(self.birthday_label)
        self.pushButton1_3 = QtWidgets.QPushButton(ChatForm)
        self.pushButton1_3.setGeometry(QtCore.QRect(150, 250, 75, 23))
        self.pushButton1_3.setObjectName("pushButton1_3")
        self.onlinestatus_label = QtWidgets.QLabel(ChatForm)
        self.onlinestatus_label.setGeometry(QtCore.QRect(660, 30, 54, 21))
        self.onlinestatus_label.setText("")
        self.onlinestatus_label.setObjectName("onlinestatus_label")
        self.style_label = QtWidgets.QLabel(ChatForm)
        self.style_label.setGeometry(QtCore.QRect(1160, 290, 211, 41))
        self.style_label.setText("")
        self.style_label.setObjectName("style_label")
        self.my_avatar_label = QtWidgets.QLabel(ChatForm)
        self.my_avatar_label.setGeometry(QtCore.QRect(20, 10, 61, 61))
        self.my_avatar_label.setText("")
        self.my_avatar_label.setScaledContents(True)
        self.my_avatar_label.setObjectName("my_avatar_label")
        self.quit_Button = QtWidgets.QPushButton(ChatForm)
        self.quit_Button.setGeometry(QtCore.QRect(10, 820, 70, 70))
        self.quit_Button.setText("")
        self.quit_Button.setObjectName("quit_Button")
        self.pushButton1_4 = QtWidgets.QPushButton(ChatForm)
        self.pushButton1_4.setGeometry(QtCore.QRect(150, 120, 75, 23))
        self.pushButton1_4.setObjectName("pushButton1_4")
        self.listWidget = QtWidgets.QListWidget(ChatForm)
        self.listWidget.setGeometry(QtCore.QRect(115, 90, 271, 751))
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.pushButton = QtWidgets.QPushButton(ChatForm)
        self.pushButton.setGeometry(QtCore.QRect(0, 160, 99, 81))
        self.pushButton.setText("")
        self.pushButton.setObjectName("pushButton")
        self.listWidget1 = QtWidgets.QListWidget(ChatForm)
        self.listWidget1.setGeometry(QtCore.QRect(115, 90, 271, 751))
        self.listWidget1.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget1.addItem(item)
        self.listWidget1_1 = ListWindow(ChatForm)
        self.listWidget1_1.setGeometry(QtCore.QRect(100, 80, 320, 800))
        self.pushButton_2 = QtWidgets.QPushButton(ChatForm)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 240, 99, 81))
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(ChatForm)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 320, 99, 81))
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(ChatForm)
        self.pushButton_4.setGeometry(QtCore.QRect(0, 400, 99, 81))
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_6 = QtWidgets.QPushButton(ChatForm)
        self.pushButton_6.setGeometry(QtCore.QRect(920, 18, 65, 48))
        self.pushButton_6.setText("")
        self.pushButton_6.setObjectName("pushButton_6")

        self.pushButton_7 = QtWidgets.QPushButton(ChatForm)
        self.pushButton_7.setGeometry(QtCore.QRect(0, 90, 99, 70))
        self.pushButton_7.setText("")
        self.pushButton_7.setObjectName("pushButton_7")

        self.retranslateUi(ChatForm)
        QtCore.QMetaObject.connectSlotsByName(ChatForm)

    def retranslateUi(self, ChatForm):
        _translate = QtCore.QCoreApplication.translate
        ChatForm.setWindowTitle(_translate("ChatForm", "Root"))
        ChatForm.setFixedSize(ChatForm.width(), ChatForm.height())
        self.pushButton1.setText(_translate("ChatForm", "10001"))
        self.pushButton1_2.setText(_translate("ChatForm", "10000"))
        self.pushButton1_3.setText(_translate("ChatForm", "10002"))
        self.pushButton1_4.setText(_translate("ChatForm", "小Root"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("ChatForm", "New Item"))
        self.listWidget.setSortingEnabled(__sortingEnabled)