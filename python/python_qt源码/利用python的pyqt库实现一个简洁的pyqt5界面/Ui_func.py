# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\bjtu\高铁3.0\func.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_high_speed(object):
    def setupUi(self, high_speed):
        high_speed.setObjectName("high_speed")
        high_speed.resize(970, 692)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(high_speed.sizePolicy().hasHeightForWidth())
        high_speed.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        high_speed.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(high_speed)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(8)
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.process_part = QtWidgets.QVBoxLayout()
        self.process_part.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.process_part.setContentsMargins(0, 2, -1, 10)
        self.process_part.setObjectName("process_part")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_pro = QtWidgets.QLabel(high_speed)
        self.label_pro.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_pro.setFont(font)
        self.label_pro.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_pro.setAlignment(QtCore.Qt.AlignCenter)
        self.label_pro.setObjectName("label_pro")
        self.horizontalLayout.addWidget(self.label_pro)
        self.info_proc = QtWidgets.QLabel(high_speed)
        self.info_proc.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.info_proc.setFont(font)
        self.info_proc.setText("")
        self.info_proc.setObjectName("info_proc")
        self.horizontalLayout.addWidget(self.info_proc)
        spacerItem = QtWidgets.QSpacerItem(100, 30, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_out = QtWidgets.QLabel(high_speed)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_out.setFont(font)
        self.label_out.setTextFormat(QtCore.Qt.AutoText)
        self.label_out.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_out.setObjectName("label_out")
        self.horizontalLayout.addWidget(self.label_out)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.info_out = QtWidgets.QTextBrowser(high_speed)
        self.info_out.setMinimumSize(QtCore.QSize(80, 40))
        self.info_out.setMaximumSize(QtCore.QSize(350, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.info_out.setFont(font)
        self.info_out.setFrameShadow(QtWidgets.QFrame.Raised)
        self.info_out.setTabStopWidth(0)
        self.info_out.setObjectName("info_out")
        self.horizontalLayout.addWidget(self.info_out)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.button_out = QtWidgets.QPushButton(high_speed)
        self.button_out.setMinimumSize(QtCore.QSize(60, 30))
        self.button_out.setMaximumSize(QtCore.QSize(100, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.button_out.setFont(font)
        self.button_out.setObjectName("button_out")
        self.horizontalLayout.addWidget(self.button_out)
        spacerItem3 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.horizontalLayout.setStretch(5, 3)
        self.process_part.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.process_part, 1, 1, 1, 1)
        self.func_part = QtWidgets.QVBoxLayout()
        self.func_part.setContentsMargins(5, 2, 0, -1)
        self.func_part.setSpacing(12)
        self.func_part.setObjectName("func_part")
        self.label_picin = QtWidgets.QLabel(high_speed)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_picin.setFont(font)
        self.label_picin.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label_picin.setMouseTracking(False)
        self.label_picin.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.label_picin.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_picin.setObjectName("label_picin")
        self.func_part.addWidget(self.label_picin)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(10)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.info_infile = QtWidgets.QTextBrowser(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.info_infile.sizePolicy().hasHeightForWidth())
        self.info_infile.setSizePolicy(sizePolicy)
        self.info_infile.setMinimumSize(QtCore.QSize(150, 30))
        self.info_infile.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.info_infile.setFont(font)
        self.info_infile.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.info_infile.setObjectName("info_infile")
        self.horizontalLayout_2.addWidget(self.info_infile)
        self.button_in = QtWidgets.QPushButton(high_speed)
        self.button_in.setMinimumSize(QtCore.QSize(60, 30))
        self.button_in.setMaximumSize(QtCore.QSize(100, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.button_in.setFont(font)
        self.button_in.setObjectName("button_in")
        self.horizontalLayout_2.addWidget(self.button_in)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.horizontalLayout_2.setStretch(0, 3)
        self.horizontalLayout_2.setStretch(1, 1)
        self.func_part.addLayout(self.horizontalLayout_2)
        spacerItem5 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.func_part.addItem(spacerItem5)
        self.button_begin = QtWidgets.QPushButton(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_begin.sizePolicy().hasHeightForWidth())
        self.button_begin.setSizePolicy(sizePolicy)
        self.button_begin.setMinimumSize(QtCore.QSize(80, 30))
        self.button_begin.setMaximumSize(QtCore.QSize(300, 60))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.button_begin.setFont(font)
        self.button_begin.setObjectName("button_begin")
        self.func_part.addWidget(self.button_begin)
        self.button_end = QtWidgets.QPushButton(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_end.sizePolicy().hasHeightForWidth())
        self.button_end.setSizePolicy(sizePolicy)
        self.button_end.setMinimumSize(QtCore.QSize(80, 30))
        self.button_end.setMaximumSize(QtCore.QSize(300, 60))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.button_end.setFont(font)
        self.button_end.setObjectName("button_end")
        self.func_part.addWidget(self.button_end)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.func_part.addItem(spacerItem6)
        self.button_pro = QtWidgets.QPushButton(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_pro.sizePolicy().hasHeightForWidth())
        self.button_pro.setSizePolicy(sizePolicy)
        self.button_pro.setMinimumSize(QtCore.QSize(80, 30))
        self.button_pro.setMaximumSize(QtCore.QSize(300, 60))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.button_pro.setFont(font)
        self.button_pro.setObjectName("button_pro")
        self.func_part.addWidget(self.button_pro)
        self.button_next = QtWidgets.QPushButton(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_next.sizePolicy().hasHeightForWidth())
        self.button_next.setSizePolicy(sizePolicy)
        self.button_next.setMinimumSize(QtCore.QSize(80, 30))
        self.button_next.setMaximumSize(QtCore.QSize(300, 60))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.button_next.setFont(font)
        self.button_next.setObjectName("button_next")
        self.func_part.addWidget(self.button_next)
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.func_part.addItem(spacerItem7)
        self.info_warn = QtWidgets.QTableWidget(high_speed)
        self.info_warn.setMaximumSize(QtCore.QSize(800, 16777215))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(8)
        font.setBold(False)
        font.setWeight(50)
        self.info_warn.setFont(font)
        self.info_warn.setObjectName("info_warn")
        self.info_warn.setColumnCount(0)
        self.info_warn.setRowCount(0)
        self.func_part.addWidget(self.info_warn)
        self.func_part.setStretch(0, 1)
        self.func_part.setStretch(1, 2)
        self.func_part.setStretch(3, 2)
        self.func_part.setStretch(4, 2)
        self.func_part.setStretch(6, 2)
        self.func_part.setStretch(7, 2)
        self.func_part.setStretch(9, 12)
        self.gridLayout.addLayout(self.func_part, 1, 3, 8, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_title = QtWidgets.QLabel(high_speed)
        self.label_title.setObjectName("label_title")
        self.horizontalLayout_6.addWidget(self.label_title)
        self.button_min = QtWidgets.QPushButton(high_speed)
        self.button_min.setMinimumSize(QtCore.QSize(0, 0))
        self.button_min.setMaximumSize(QtCore.QSize(40, 70))
        self.button_min.setText("")
        self.button_min.setIconSize(QtCore.QSize(16, 16))
        self.button_min.setObjectName("button_min")
        self.horizontalLayout_6.addWidget(self.button_min)
        self.button_max = QtWidgets.QPushButton(high_speed)
        self.button_max.setMinimumSize(QtCore.QSize(0, 0))
        self.button_max.setMaximumSize(QtCore.QSize(40, 70))
        self.button_max.setText("")
        self.button_max.setIconSize(QtCore.QSize(16, 16))
        self.button_max.setObjectName("button_max")
        self.horizontalLayout_6.addWidget(self.button_max)
        self.button_close = QtWidgets.QPushButton(high_speed)
        self.button_close.setMinimumSize(QtCore.QSize(0, 0))
        self.button_close.setMaximumSize(QtCore.QSize(40, 70))
        self.button_close.setText("")
        self.button_close.setIconSize(QtCore.QSize(16, 16))
        self.button_close.setObjectName("button_close")
        self.horizontalLayout_6.addWidget(self.button_close)
        self.verticalLayout_2.addLayout(self.horizontalLayout_6)
        self.verticalLayout_2.setStretch(0, 5)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 3)
        self.detection_part = QtWidgets.QHBoxLayout()
        self.detection_part.setSpacing(0)
        self.detection_part.setObjectName("detection_part")
        self.label_monitor = QtWidgets.QLabel(high_speed)
        self.label_monitor.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_monitor.setFont(font)
        self.label_monitor.setAlignment(QtCore.Qt.AlignCenter)
        self.label_monitor.setObjectName("label_monitor")
        self.detection_part.addWidget(self.label_monitor)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 5, -1, 5)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.button_rightup = QtWidgets.QPushButton(high_speed)
        self.button_rightup.setMinimumSize(QtCore.QSize(0, 30))
        self.button_rightup.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_rightup.setFont(font)
        self.button_rightup.setObjectName("button_rightup")
        self.horizontalLayout_4.addWidget(self.button_rightup)
        self.button_leftup = QtWidgets.QPushButton(high_speed)
        self.button_leftup.setMinimumSize(QtCore.QSize(0, 30))
        self.button_leftup.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_leftup.setFont(font)
        self.button_leftup.setObjectName("button_leftup")
        self.horizontalLayout_4.addWidget(self.button_leftup)
        self.button_rightdown = QtWidgets.QPushButton(high_speed)
        self.button_rightdown.setMinimumSize(QtCore.QSize(0, 30))
        self.button_rightdown.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_rightdown.setFont(font)
        self.button_rightdown.setObjectName("button_rightdown")
        self.horizontalLayout_4.addWidget(self.button_rightdown)
        self.button_leftdown = QtWidgets.QPushButton(high_speed)
        self.button_leftdown.setMinimumSize(QtCore.QSize(0, 30))
        self.button_leftdown.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_leftdown.setFont(font)
        self.button_leftdown.setObjectName("button_leftdown")
        self.horizontalLayout_4.addWidget(self.button_leftdown)
        self.button_basemid = QtWidgets.QPushButton(high_speed)
        self.button_basemid.setMinimumSize(QtCore.QSize(0, 30))
        self.button_basemid.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_basemid.setFont(font)
        self.button_basemid.setObjectName("button_basemid")
        self.horizontalLayout_4.addWidget(self.button_basemid)
        self.button_baseinright = QtWidgets.QPushButton(high_speed)
        self.button_baseinright.setMinimumSize(QtCore.QSize(0, 30))
        self.button_baseinright.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_baseinright.setFont(font)
        self.button_baseinright.setObjectName("button_baseinright")
        self.horizontalLayout_4.addWidget(self.button_baseinright)
        self.button_baseinleft = QtWidgets.QPushButton(high_speed)
        self.button_baseinleft.setMinimumSize(QtCore.QSize(0, 30))
        self.button_baseinleft.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_baseinleft.setFont(font)
        self.button_baseinleft.setObjectName("button_baseinleft")
        self.horizontalLayout_4.addWidget(self.button_baseinleft)
        self.button_baseoutright = QtWidgets.QPushButton(high_speed)
        self.button_baseoutright.setMinimumSize(QtCore.QSize(0, 30))
        self.button_baseoutright.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_baseoutright.setFont(font)
        self.button_baseoutright.setObjectName("button_baseoutright")
        self.horizontalLayout_4.addWidget(self.button_baseoutright)
        self.button_baseoutleft = QtWidgets.QPushButton(high_speed)
        self.button_baseoutleft.setMinimumSize(QtCore.QSize(0, 30))
        self.button_baseoutleft.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.button_baseoutleft.setFont(font)
        self.button_baseoutleft.setObjectName("button_baseoutleft")
        self.horizontalLayout_4.addWidget(self.button_baseoutleft)
        self.detection_part.addLayout(self.horizontalLayout_4)
        self.detection_part.setStretch(0, 1)
        self.detection_part.setStretch(1, 8)
        self.gridLayout.addLayout(self.detection_part, 5, 1, 1, 1)
        self.label_picall = QtWidgets.QLabel(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_picall.sizePolicy().hasHeightForWidth())
        self.label_picall.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_picall.setFont(font)
        self.label_picall.setLineWidth(0)
        self.label_picall.setScaledContents(True)
        self.label_picall.setAlignment(QtCore.Qt.AlignCenter)
        self.label_picall.setObjectName("label_picall")
        self.gridLayout.addWidget(self.label_picall, 2, 1, 1, 1)
        self.label_state = QtWidgets.QLabel(high_speed)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_state.setFont(font)
        self.label_state.setText("")
        self.label_state.setObjectName("label_state")
        self.gridLayout.addWidget(self.label_state, 9, 1, 1, 3)
        self.ver_line = QtWidgets.QFrame(high_speed)
        self.ver_line.setMinimumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.ver_line.setFont(font)
        self.ver_line.setStyleSheet("color: rgb(102, 102, 102);")
        self.ver_line.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ver_line.setLineWidth(5)
        self.ver_line.setFrameShape(QtWidgets.QFrame.VLine)
        self.ver_line.setObjectName("ver_line")
        self.gridLayout.addWidget(self.ver_line, 1, 2, 7, 1)
        self.pass_part = QtWidgets.QHBoxLayout()
        self.pass_part.setSpacing(0)
        self.pass_part.setObjectName("pass_part")
        self.label_passrail = QtWidgets.QLabel(high_speed)
        self.label_passrail.setMaximumSize(QtCore.QSize(100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_passrail.setFont(font)
        self.label_passrail.setAlignment(QtCore.Qt.AlignCenter)
        self.label_passrail.setObjectName("label_passrail")
        self.pass_part.addWidget(self.label_passrail)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setContentsMargins(-1, 5, -1, 5)
        self.horizontalLayout_9.setSpacing(0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.passseq_1 = QtWidgets.QPushButton(high_speed)
        self.passseq_1.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_1.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_1.setFont(font)
        self.passseq_1.setObjectName("passseq_1")
        self.verticalLayout_3.addWidget(self.passseq_1)
        self.passseq_9 = QtWidgets.QPushButton(high_speed)
        self.passseq_9.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_9.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_9.setFont(font)
        self.passseq_9.setObjectName("passseq_9")
        self.verticalLayout_3.addWidget(self.passseq_9)
        self.horizontalLayout_9.addLayout(self.verticalLayout_3)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setSpacing(6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.passseq_2 = QtWidgets.QPushButton(high_speed)
        self.passseq_2.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_2.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_2.setFont(font)
        self.passseq_2.setObjectName("passseq_2")
        self.verticalLayout_9.addWidget(self.passseq_2)
        self.passseq_10 = QtWidgets.QPushButton(high_speed)
        self.passseq_10.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_10.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_10.setFont(font)
        self.passseq_10.setObjectName("passseq_10")
        self.verticalLayout_9.addWidget(self.passseq_10)
        self.horizontalLayout_9.addLayout(self.verticalLayout_9)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSpacing(6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.passseq_3 = QtWidgets.QPushButton(high_speed)
        self.passseq_3.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_3.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_3.setFont(font)
        self.passseq_3.setObjectName("passseq_3")
        self.verticalLayout_7.addWidget(self.passseq_3)
        self.passseq_11 = QtWidgets.QPushButton(high_speed)
        self.passseq_11.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_11.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_11.setFont(font)
        self.passseq_11.setObjectName("passseq_11")
        self.verticalLayout_7.addWidget(self.passseq_11)
        self.horizontalLayout_9.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setSpacing(6)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.passseq_4 = QtWidgets.QPushButton(high_speed)
        self.passseq_4.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_4.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_4.setFont(font)
        self.passseq_4.setObjectName("passseq_4")
        self.verticalLayout_8.addWidget(self.passseq_4)
        self.passseq_12 = QtWidgets.QPushButton(high_speed)
        self.passseq_12.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_12.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_12.setFont(font)
        self.passseq_12.setObjectName("passseq_12")
        self.verticalLayout_8.addWidget(self.passseq_12)
        self.horizontalLayout_9.addLayout(self.verticalLayout_8)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSpacing(6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.passseq_5 = QtWidgets.QPushButton(high_speed)
        self.passseq_5.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_5.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_5.setFont(font)
        self.passseq_5.setObjectName("passseq_5")
        self.verticalLayout_6.addWidget(self.passseq_5)
        self.passseq_13 = QtWidgets.QPushButton(high_speed)
        self.passseq_13.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_13.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_13.setFont(font)
        self.passseq_13.setObjectName("passseq_13")
        self.verticalLayout_6.addWidget(self.passseq_13)
        self.horizontalLayout_9.addLayout(self.verticalLayout_6)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setSpacing(6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.passseq_6 = QtWidgets.QPushButton(high_speed)
        self.passseq_6.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_6.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_6.setFont(font)
        self.passseq_6.setObjectName("passseq_6")
        self.verticalLayout_5.addWidget(self.passseq_6)
        self.passseq_14 = QtWidgets.QPushButton(high_speed)
        self.passseq_14.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_14.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_14.setFont(font)
        self.passseq_14.setObjectName("passseq_14")
        self.verticalLayout_5.addWidget(self.passseq_14)
        self.horizontalLayout_9.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.passseq_7 = QtWidgets.QPushButton(high_speed)
        self.passseq_7.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_7.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_7.setFont(font)
        self.passseq_7.setObjectName("passseq_7")
        self.verticalLayout_4.addWidget(self.passseq_7)
        self.passseq_15 = QtWidgets.QPushButton(high_speed)
        self.passseq_15.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_15.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_15.setFont(font)
        self.passseq_15.setObjectName("passseq_15")
        self.verticalLayout_4.addWidget(self.passseq_15)
        self.horizontalLayout_9.addLayout(self.verticalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.passseq_8 = QtWidgets.QPushButton(high_speed)
        self.passseq_8.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_8.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_8.setFont(font)
        self.passseq_8.setObjectName("passseq_8")
        self.verticalLayout.addWidget(self.passseq_8)
        self.passseq_16 = QtWidgets.QPushButton(high_speed)
        self.passseq_16.setMinimumSize(QtCore.QSize(0, 30))
        self.passseq_16.setMaximumSize(QtCore.QSize(16777215, 45))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.passseq_16.setFont(font)
        self.passseq_16.setObjectName("passseq_16")
        self.verticalLayout.addWidget(self.passseq_16)
        self.horizontalLayout_9.addLayout(self.verticalLayout)
        self.pass_part.addLayout(self.horizontalLayout_9)
        self.pass_part.setStretch(0, 1)
        self.pass_part.setStretch(1, 9)
        self.gridLayout.addLayout(self.pass_part, 6, 1, 1, 1)
        self.railnum_part = QtWidgets.QHBoxLayout()
        self.railnum_part.setSpacing(0)
        self.railnum_part.setObjectName("railnum_part")
        self.label_set = QtWidgets.QLabel(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_set.sizePolicy().hasHeightForWidth())
        self.label_set.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_set.setFont(font)
        self.label_set.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_set.setObjectName("label_set")
        self.railnum_part.addWidget(self.label_set)
        self.comboBox_setnum = QtWidgets.QComboBox(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_setnum.sizePolicy().hasHeightForWidth())
        self.comboBox_setnum.setSizePolicy(sizePolicy)
        self.comboBox_setnum.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_setnum.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.comboBox_setnum.setFont(font)
        self.comboBox_setnum.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox_setnum.setAutoFillBackground(False)
        self.comboBox_setnum.setStyleSheet("selection-background-color: rgb(30, 30, 30);")
        self.comboBox_setnum.setFrame(False)
        self.comboBox_setnum.setObjectName("comboBox_setnum")
        self.railnum_part.addWidget(self.comboBox_setnum)
        self.label_rail = QtWidgets.QLabel(high_speed)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_rail.setFont(font)
        self.label_rail.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_rail.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_rail.setObjectName("label_rail")
        self.railnum_part.addWidget(self.label_rail)
        self.comboBox_railnum = QtWidgets.QComboBox(high_speed)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_railnum.sizePolicy().hasHeightForWidth())
        self.comboBox_railnum.setSizePolicy(sizePolicy)
        self.comboBox_railnum.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox_railnum.setMaximumSize(QtCore.QSize(200, 40))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.comboBox_railnum.setFont(font)
        self.comboBox_railnum.setStyleSheet("selection-background-color: rgb(30, 30, 30);")
        self.comboBox_railnum.setFrame(False)
        self.comboBox_railnum.setObjectName("comboBox_railnum")
        self.railnum_part.addWidget(self.comboBox_railnum)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.railnum_part.addItem(spacerItem8)
        self.gridLayout.addLayout(self.railnum_part, 7, 1, 1, 1)
        # self.label_pic = QtWidgets.QLabel(high_speed)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.label_pic.sizePolicy().hasHeightForWidth())
        # self.label_pic.setSizePolicy(sizePolicy)
        # self.label_pic.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_pic.setObjectName("label_pic")
        # self.gridLayout.addWidget(self.label_pic, 3, 1, 1, 1)
        self.gridLayout.setColumnStretch(1, 7)
        self.gridLayout.setColumnStretch(3, 3)
        self.gridLayout.setRowStretch(0, 2)
        self.gridLayout.setRowStretch(1, 1)
        self.gridLayout.setRowStretch(2, 6)
        self.gridLayout.setRowStretch(3, 20)
        self.gridLayout.setRowStretch(5, 2)
        self.gridLayout.setRowStretch(6, 4)
        self.gridLayout.setRowStretch(7, 1)
        self.gridLayout.setRowStretch(9, 1)

        self.retranslateUi(high_speed)
        self.comboBox_setnum.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(high_speed)

    def retranslateUi(self, high_speed):
        _translate = QtCore.QCoreApplication.translate
        high_speed.setWindowTitle(_translate("high_speed", "动车组故障图像检测系统"))
        self.label_pro.setText(_translate("high_speed", "看图进度"))
        self.label_out.setText(_translate("high_speed", "导出结果文件"))
        self.info_out.setHtml(_translate("high_speed", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'宋体\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'SimSun\'; font-size:9pt;\"><br /></p></body></html>"))
        self.button_out.setText(_translate("high_speed", "导出"))
        self.label_picin.setText(_translate("high_speed", "图片"))
        self.button_in.setText(_translate("high_speed", "导入"))
        self.button_begin.setText(_translate("high_speed", "开始检测"))
        self.button_end.setText(_translate("high_speed", "结束检测"))
        self.button_pro.setText(_translate("high_speed", "上一张"))
        self.button_next.setText(_translate("high_speed", "下一张"))
        self.label_title.setText(_translate("high_speed", "    动车组故障图像检测系统"))
        self.label_monitor.setText(_translate("high_speed", "监测部位"))
        self.button_rightup.setText(_translate("high_speed", "右侧上"))
        self.button_leftup.setText(_translate("high_speed", "左侧上"))
        self.button_rightdown.setText(_translate("high_speed", "右侧下"))
        self.button_leftdown.setText(_translate("high_speed", "左侧下"))
        self.button_basemid.setText(_translate("high_speed", "底中"))
        self.button_baseinright.setText(_translate("high_speed", "底内右"))
        self.button_baseinleft.setText(_translate("high_speed", "底内左"))
        self.button_baseoutright.setText(_translate("high_speed", "底外右"))
        self.button_baseoutleft.setText(_translate("high_speed", "底外左"))
        # self.label_picall.setText(_translate("high_speed", "all"))
        self.label_passrail.setText(_translate("high_speed", "过车序号"))
        self.passseq_1.setText(_translate("high_speed", "1"))
        self.passseq_9.setText(_translate("high_speed", "9"))
        self.passseq_2.setText(_translate("high_speed", "2"))
        self.passseq_10.setText(_translate("high_speed", "10"))
        self.passseq_3.setText(_translate("high_speed", "3"))
        self.passseq_11.setText(_translate("high_speed", "11"))
        self.passseq_4.setText(_translate("high_speed", "4"))
        self.passseq_12.setText(_translate("high_speed", "12"))
        self.passseq_5.setText(_translate("high_speed", "5"))
        self.passseq_13.setText(_translate("high_speed", "13"))
        self.passseq_6.setText(_translate("high_speed", "6"))
        self.passseq_14.setText(_translate("high_speed", "14"))
        self.passseq_7.setText(_translate("high_speed", "7"))
        self.passseq_15.setText(_translate("high_speed", "15"))
        self.passseq_8.setText(_translate("high_speed", "8"))
        self.passseq_16.setText(_translate("high_speed", "16"))
        self.label_set.setText(_translate("high_speed", "车组号"))
        self.label_rail.setText(_translate("high_speed", "车辆号"))
        # self.label_pic.setText(_translate("high_speed", "result"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    high_speed = QtWidgets.QWidget()
    ui = Ui_high_speed()
    ui.setupUi(high_speed)
    high_speed.show()
    sys.exit(app.exec_())
