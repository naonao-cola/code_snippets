# coding:utf-8

from PySide2.QtCore import Qt
from PySide2.QtGui import QResizeEvent
from PySide2.QtWidgets import QWidget
from win32.lib import win32con
from win32.win32api import SendMessage
from win32.win32gui import ReleaseCapture

from .title_bar_buttons import MaximizeButton, ThreeStateToolButton


class TitleBar(QWidget):

    def __init__(self, skinName, parent):
        super().__init__(parent)
        self.setMouseTracking(True)  # 否则只有左键时才能获取鼠标坐标
        self.skinName = skinName

        self.minBtn = ThreeStateToolButton(
            {'normal': f'{self.skinName}/title_bar/最小化按钮_normal_57_40.png',
             'hover': f'{self.skinName}/title_bar/最小化按钮_hover_57_40.png',
             'pressed': f'{self.skinName}/title_bar/最小化按钮_pressed_57_40.png'}, (28, 28), self)
        self.closeBtn = ThreeStateToolButton(
            {'normal': f'{self.skinName}/title_bar/关闭按钮_normal_57_40.png',
             'hover': f'{self.skinName}/title_bar/关闭按钮_hover_57_40.png',
             'pressed': f'{self.skinName}/title_bar/关闭按钮_pressed_57_40.png'}, (28, 28), self)

        self.maxBtn = MaximizeButton(skinName,self)

        self.__initWidget()

    def __initWidget(self):
        """ initialize all widgets """
        self.resize(1360, 28)
        self.setFixedHeight(28)
        self.setAttribute(Qt.WA_StyledBackground)
        self.__setQss()

        # connect signal to slot
        self.minBtn.clicked.connect(self.window().showMinimized)
        self.maxBtn.clicked.connect(self.__toggleMaxState)
        self.closeBtn.clicked.connect(self.window().close)

    def resizeEvent(self, e: QResizeEvent):
        """ Move the buttons """
        self.closeBtn.move(self.width() - 28, 0)
        self.maxBtn.move(self.width() - 2 * 28, 0)
        self.minBtn.move(self.width() - 3 * 28, 0)

    def mouseDoubleClickEvent(self, event):
        """ Toggles the maximization state of the window """
        self.__toggleMaxState()

    def mousePressEvent(self, event):
        """ Move the window """
        if not self.__isPointInDragRegion(event.pos()):
            return

        ReleaseCapture()
        SendMessage(self.window().winId(), win32con.WM_SYSCOMMAND,
                    win32con.SC_MOVE + win32con.HTCAPTION, 0)
        event.ignore()

    def __toggleMaxState(self):
        """ Toggles the maximization state of the window and change icon """
        if self.window().isMaximized():
            self.window().showNormal()
            # change the icon of maxBtn
            self.maxBtn.setMaxState(False)
        else:
            self.window().showMaximized()
            self.maxBtn.setMaxState(True)

    def __isPointInDragRegion(self, pos) -> bool:
        """ Check whether the pressed point belongs to the area where dragging is allowed """
        right = self.width() - 28 * 4 if self.minBtn.isVisible() else self.width() - 28
        return (0 < pos.x() < right)

    def __setQss(self):
        self.setStyleSheet("QWidget {\n"
                           "            background-color: transparent\n"
                           "            }\n"
                           "            \n"
                           "            QToolButton{\n"
                           "            background-color: transparent;\n"
                           "            border: none;\n"
                           "            margin: 0px;\n"
                           "            }")

class MainWindowTitleBar(TitleBar):
    def __init__(self, skinName, parent):
        super().__init__(skinName,parent)

        self.setBtn = ThreeStateToolButton(
            {'normal': f'{self.skinName}/title_bar/设置_normal_40_40.png',
             'hover': f'{self.skinName}/title_bar/设置_hover_40_40.png',
             'pressed': f'{self.skinName}/title_bar/设置_pressed_40_40.png'}, (28, 28), self)

    def resizeEvent(self, e: QResizeEvent):
        super().resizeEvent(e)
        self.setBtn.move(self.width() - 4 * 28, 0)

    def __isPointInDragRegion(self, pos) -> bool:
        """ Check whether the pressed point belongs to the area where dragging is allowed """
        right = self.width() - 28 * 4 if self.minBtn.isVisible() else self.width() - 28
        return (0 < pos.x() < right)