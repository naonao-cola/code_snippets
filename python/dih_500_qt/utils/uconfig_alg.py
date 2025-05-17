import os
import json
from utils.ufile_func import *
from PyQt5.QtWidgets import QMenu, QAction, QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox

from ui.ualg_cfg import Ui_Form
# 算法配置文件
# 对应于 config/algo/alg_cfg 文件夹下的 JSON 文件


class AlgConfigEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 连接按钮信号
        self.ui.pushButton.clicked.connect(self.load_config)
        self.ui.pushButton_2.clicked.connect(self.save_config)
        self.config_path = get_config_path("algo", "alg_cfg")

    def load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # 清晰度json
            clarity_json = find_in_nested(config, "001", "type_id")
            self.ui.lineEdit.setText(
                str(clarity_json["algo_params"][0]["param"]["detect_flag"]))
            self.ui.lineEdit_2.setText(
                str(clarity_json["algo_params"][0]["param"]["model_idx_1"]))
            self.ui.lineEdit_3.setText(
                str(clarity_json["algo_params"][0]["param"]["model_idx_2"]))
            self.ui.lineEdit_4.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_near"]))
            self.ui.lineEdit_5.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_far"]))
            self.ui.lineEdit_6.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_clear"]))
            self.ui.lineEdit_7.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_rbc"]))
            self.ui.lineEdit_18.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_coarse_focus"]))
            self.ui.lineEdit_9.setText(
                str(clarity_json["algo_params"][0]["param"]["far_near_change_view"]))
            self.ui.lineEdit_10.setText(
                str(clarity_json["algo_params"][0]["param"]["coarse_category_num"]))
            self.ui.lineEdit_11.setText(
                str(clarity_json["algo_params"][0]["param"]["coarse_clear"]))

            # rbc 的json
            rbc_json = find_in_nested(config, "003", "type_id")
            self.ui.lineEdit_12.setText(
                str(rbc_json["algo_params"][0]["param"]["detect_model_idx1"]))
            self.ui.lineEdit_13.setText(
                str(rbc_json["algo_params"][0]["param"]["detect_model_idx2"]))
            self.ui.lineEdit_14.setText(
                str(rbc_json["algo_params"][0]["param"]["detect_model_idx3"]))
            self.ui.lineEdit_15.setText(
                str(rbc_json["algo_params"][0]["param"]["seg_model_idx"]))
            self.ui.lineEdit_16.setText(
                str(rbc_json["algo_params"][0]["param"]["img_count"]))
            self.ui.lineEdit_17.setText(
                str(rbc_json["algo_params"][0]["param"]["dilution_ratio"]))
            self.ui.lineEdit_18.setText(
                str(rbc_json["algo_params"][0]["param"]["pix_value"]))
            self.ui.lineEdit_19.setText(
                str(rbc_json["algo_params"][0]["param"]["coarse_category_num"]))
            self.ui.lineEdit_20.setText(
                str(rbc_json["algo_params"][0]["param"]["flow_high"]))

            # wbc 的json
            wbc_json = find_in_nested(config, "004", "type_id")
            self.ui.lineEdit_21.setText(
                str(wbc_json["algo_params"][0]["param"]["detect_model_idx1"]))
            self.ui.lineEdit_22.setText(
                str(wbc_json["algo_params"][0]["param"]["detect_model_idx2"]))
            self.ui.lineEdit_23.setText(
                str(wbc_json["algo_params"][0]["param"]["img_count"]))
            self.ui.lineEdit_24.setText(
                str(wbc_json["algo_params"][0]["param"]["dilution_ratio"]))
            self.ui.lineEdit_25.setText(
                str(wbc_json["algo_params"][0]["param"]["pix_value"]))
            self.ui.lineEdit_26.setText(
                str(wbc_json["algo_params"][0]["param"]["flow_high"]))

            # baso 的json
            baso_json = find_in_nested(config, "002", "type_id")
            self.ui.lineEdit_27.setText(
                str(baso_json["algo_params"][0]["param"]["detect_model_idx1"]))
            self.ui.lineEdit_28.setText(
                str(baso_json["algo_params"][0]["param"]["img_count"]))
            self.ui.lineEdit_29.setText(
                str(baso_json["algo_params"][0]["param"]["dilution_ratio"]))
            self.ui.lineEdit_30.setText(
                str(baso_json["algo_params"][0]["param"]["pix_value"]))
            self.ui.lineEdit_31.setText(
                str(baso_json["algo_params"][0]["param"]["flow_high"]))
            QMessageBox.information(self, "成功", f"读取成功 {self.config_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")

    def save_config(self):
        try:
            config = [
                {
                    "type_id": "001",
                    "type_name": "Example",
                    "pre_algo_params": [],
                    "algo_params": [
                        {
                            "algo_name": "Clarity",
                            "algo_index": 1,
                            "param": {
                                "detect_flag": int(self.ui.lineEdit.text()),
                                "model_idx_1": int(self.ui.lineEdit_2.text()),
                                "model_idx_2": int(self.ui.lineEdit_3.text()),
                                "far_near_near": float(self.ui.lineEdit_4.text()),
                                "far_near_far": float(self.ui.lineEdit_5.text()),
                                "far_near_clear": float(self.ui.lineEdit_6.text()),
                                "far_near_rbc": float(self.ui.lineEdit_7.text()),
                                "far_near_coarse_focus": float(self.ui.lineEdit_8.text()),
                                "far_near_change_view": float(self.ui.lineEdit_9.text()),
                                "coarse_category_num": int(self.ui.lineEdit_10.text()),
                                "coarse_clear": float(self.ui.lineEdit_11.text()),
                            }
                        }
                    ]
                },
                {
                    "type_id": "002",
                    "type_name": "Example",
                    "pre_algo_params": [],
                    "algo_params": [
                        {
                            "algo_name": "BasoAlg",
                            "algo_index": 1,
                            "param": {
                                "detect_model_idx1": int(self.ui.lineEdit_27.text()),
                                "img_count": int(self.ui.lineEdit_28.text()),
                                "dilution_ratio": int(self.ui.lineEdit_29.text()),
                                "pix_value": float(self.ui.lineEdit_30.text()),
                                "flow_high": int(self.ui.lineEdit_31.text()),
                            }
                        }
                    ]
                },
                {
                    "type_id": "003",
                    "type_name": "Example",
                    "pre_algo_params": [],
                    "algo_params": [
                        {
                            "algo_name": "RbcAlg",
                            "algo_index": 1,
                            "param": {
                                "detect_model_idx1": int(self.ui.lineEdit_12.text()),
                                "detect_model_idx2": int(self.ui.lineEdit_13.text()),
                                "detect_model_idx3": int(self.ui.lineEdit_14.text()),
                                "seg_model_idx": int(self.ui.lineEdit_15.text()),
                                "img_count": int(self.ui.lineEdit_16.text()),
                                "dilution_ratio": int(self.ui.lineEdit_17.text()),
                                "pix_value": float(self.ui.lineEdit_18.text()),
                                "coarse_category_num": int(self.ui.lineEdit_19.text()),
                                "flow_high": int(self.ui.lineEdit_20.text()),
                            }
                        }
                    ]
                },
                {
                    "type_id": "004",
                    "type_name": "Example",
                    "pre_algo_params": [],
                    "algo_params": [
                        {
                            "algo_name": "WbcAlg",
                            "algo_index": 1,
                            "param": {
                                "detect_model_idx1": int(self.ui.lineEdit_21.text()),
                                "detect_model_idx2": int(self.ui.lineEdit_22.text()),
                                "img_count": int(self.ui.lineEdit_23.text()),
                                "dilution_ratio": int(self.ui.lineEdit_24.text()),
                                "pix_value": float(self.ui.lineEdit_25.text()),
                                "flow_high": int(self.ui.lineEdit_26.text()),
                            }
                        }
                    ]
                }
            ]
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "成功", "配置已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")
