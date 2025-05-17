#!/usr/bin/python
# -*- coding: UTF-8 -*-

def apply_windows_style():
    # 应用 Windows 风格的样式表
    style_sheet = """
            QMainWindow {
                background-color: #f0f0f0;
                color: #000000;
            }

            QTreeWidget {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                selection-background-color: #0000ff;
                selection-color: #ffffff;
            }

            QTreeWidget::item {
                padding: 4px;
                border: 1px solid transparent;
            }

            QTreeWidget::item:hover {
                background-color: #e0e0e0;
                border: 1px solid #a0c0c0;
            }

            QTreeWidget::item:selected {
                background-color: #0000ff;
                color: #ffffff;
                border: 1px solid #808080;
            }

            QLabel {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
            }

            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
            }

            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: #ffffff;
            }

            QTabWidget::tab-bar {
                alignment: left;
            }

            QTabBar::tab {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                padding: 5px 10px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom-color: #ffffff;
            }

            QSplitter::handle {
                background-color: #c0c0c0;
            }

            QSplitter::handle:hover {
                background-color: #a0a0a0;
            }

            QMenu {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
            }

            QMenu::item {
                padding: 4px 20px 4px 20px;
            }

            QMenu::item:selected {
                background-color: #e0e0e0;
            }
        """
    return style_sheet

