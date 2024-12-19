import os
import pandas as pd
import time
import cv2
import torch
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from models.experimental import attempt_load
from utils.general import (non_max_suppression, scale_coords, check_img_size, box_iou)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from tool.img_cut import img_cut, mapping_box


class Model():
    def __init__(self,model_json):
        model_path = model_json['model']
        self.device = select_device('0')
        self.paramers = model_json["paramers"]
        self.model = attempt_load(model_path, self.device)
        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names


    def data_proecess(self, model, img, imgsz, conf_thres, iou_thres):
        imgsz = check_img_size(imgsz, s=model.stride.max())
        img0 = img
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Convert
        img = torch.from_numpy(img).to(self.device).float()
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres)
        det = pred[0]
        # 不同类别之间也要做nms过滤
        det = det.cpu().numpy()
        keep_index = self.nms(det, iou_thres)
        det = det[keep_index, :]
        det = torch.from_numpy(det).to(self.device).float()

        # 训练时wh经过resize,需要转换回去
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        pred = det.cpu().numpy()

        return pred


    def get_final_code(self, pred_result, priority, threshold):
        temp_result = pred_result.copy()
        pred_result = pred_result.reset_index(drop=True)

        if pred_result.shape[0] == 1:
            final = pred_result.loc[0]
            return final
        else:
            for key, value in threshold.items():
                temp = temp_result[temp_result['code'] == key]
                temp_result = temp_result[temp_result['code'] != key]
                temp = temp[temp['confidence'] >= float(value)]
                temp_result = pd.concat([temp_result, temp], axis=0).reset_index(drop=True)

            if temp_result.shape[0] == 1:
                final = temp_result.loc[0]
            elif temp_result.shape[0] > 1:
                final = self.priority(pred_result, priority)
            else:
                final = pred_result.sort_values(by='confidence', ascending=False).loc[0]

            return final


    def priority(self, pred_result, priority):
        # 优先级排序
        for order in priority:
            if order in list(pred_result['code']):
                temp = pred_result[pred_result["code"] == order]
                if temp.shape[0] == 1:
                    temp = temp.iloc[0]
                else:
                    temp = temp.sort_values(by='confidence', ascending=False).iloc[0]

                return temp
            else:
                temp = pred_result.sort_values(by='confidence', ascending=False).iloc[0]
                return temp


    def nms(self, bboxs, thresh):
        x1 = bboxs[:, 0]
        y1 = bboxs[:, 1]
        x2 = bboxs[:, 2]
        y2 = bboxs[:, 3]
        order = bboxs[:, 4].argsort()[::-1]
        # print(order)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            over = (w * h) / (area[i] + area[order[1:]] - w * h)
            index = np.where(over <= thresh)[0]
            order = order[index + 1]
        return keep


    def creat_dict(self):
        final_dict = {}
        final_dict.setdefault("img_cls", "")
        final_dict.setdefault("img_box", "")
        final_dict.setdefault("img_score", "")

        result = {}
        result.setdefault("img_cls", [])
        result.setdefault("img_box", [])
        result.setdefault("img_score", [])
        result.setdefault("uid", [])
        result.setdefault("gid", [])
        result.setdefault("defect", [])
        result.setdefault("type", [])
        result.setdefault("savepath", [])
        result.setdefault("final", final_dict)

        Final_result = {}
        result.setdefault("img_cls", [])
        result.setdefault("img_box", [])
        result.setdefault("img_score", [])
        result.setdefault("uid", [])
        result.setdefault("gid", [])
        result.setdefault("defect", [])
        result.setdefault("type", [])
        result.setdefault("savepath", [])

        rr = {}
        rr.setdefault("status", 0)
        rr.setdefault("message", 0)
        rr.setdefault("result", [result, Final_result])

        return rr


    def dump_result(self, pattern, pred_result, final, status, uid, gid, defect, img_type, save_path):
        code = list(pred_result['code'])
        pred_result = pred_result.reset_index(drop=True)
        score = list(pred_result['confidence'])
        boxes = []
        for index, row in enumerate(pred_result.index):
            box = [pred_result.loc[index]["xmin"], pred_result.loc[index]["ymin"], pred_result.loc[index]["xmax"],
                   pred_result.loc[index]["ymin"], pred_result.loc[index]["xmax"], pred_result.loc[index]["ymax"],
                   pred_result.loc[index]["xmin"], pred_result.loc[index]["ymax"]]
            boxes.append(box)

        final_box = [final["xmin"], final["ymin"], final["xmax"], final["ymin"], final["xmax"], final["ymax"], final["xmin"], final["ymax"]]

        pattern["status"] = status
        pattern["message"] = "Success"
        pattern["result"][0]["img_cls"] = list(code)
        pattern["result"][0]["img_score"] = list(score)
        pattern["result"][0]["img_box"] = boxes
        pattern["result"][0]["uid"] = uid
        pattern["result"][0]["gid"] = gid
        pattern["result"][0]["defect"] = defect
        pattern["result"][0]["type"] = img_type
        pattern["result"][0]["savepath"] = save_path
        pattern["result"][0]["final"]["img_cls"] = [final["code"]]
        pattern["result"][0]["final"]["img_score"] = [final["confidence"]]
        pattern["result"][0]["final"]["img_box"] = [final_box]

        pattern["result"][1]["img_cls"] = [final["code"]]
        pattern["result"][1]["img_box"] = [final_box]
        pattern["result"][1]["img_score"] = [final["confidence"]]
        pattern["result"][1]["gid"] = gid
        pattern["result"][1]["defect"] = defect
        pattern["result"][1]["type"] = "Final"
        pattern["result"][1]["savepath"] = save_path


    def pred_filter(self, pred_result):
        pred_result = pred_result[pred_result['code'] != 'ECHO'].reset_index(drop=True)

        if 'P306' in list(pred_result['code']):
            P306 = pred_result[pred_result['code'] == 'P306'].reset_index(drop=True)
            for row in P306.iterrows():
                xmin, ymin, xmax, ymax = row[1]['xmin'], row[1]['ymin'], row[1]['xmax'], row[1]['ymax']
                area = (xmax - xmin) * (ymax - ymin)
                if area < 180:
                    P306.drop(row[0])
                    continue
            temp_2 = pred_result[pred_result['code'].isin(['P304', 'P302', 'B510A', 'B510B'])]
            if temp_2['confidence'].max() > 0.5:
                pred_result = pred_result[pred_result['code'] != 'P306']

        if "obscure" in list(pred_result['code']):
            obscure =  pred_result[pred_result['code'].isin(['obscure'])]
            pred_result = pred_result[~pred_result['code'].isin(['obscure'])]
            obscure = obscure[obscure['confidence'] > 0.5]
            pred_result = pd.concat([obscure, pred_result], axis=0).reset_index(drop=True)

        if "P501" in list(pred_result['code']):
            P501 =  pred_result[pred_result['code'].isin(['P501'])]
            pred_result = pred_result[~pred_result['code'].isin(['P501'])]
            P501 = P501[P501['confidence'] > 0.6]
            pred_result = pd.concat([P501, pred_result], axis=0).reset_index(drop=True)

        if "B510A" in list(pred_result['code']):
            B501A =  pred_result[pred_result['code'].isin(['B501A'])]
            pred_result = pred_result[~pred_result['code'].isin(['B501A'])]
            B501A = B501A[B501A['confidence'] > 0.3]
            pred_result = pd.concat([B501A, pred_result], axis=0).reset_index(drop=True)

        if "B510B" in list(pred_result['code']):
            B501B =  pred_result[pred_result['code'].isin(['B501B'])]
            pred_result = pred_result[~pred_result['code'].isin(['B501B'])]
            B501B = B501B[B501B['confidence'] > 0.3]
            pred_result = pd.concat([B501B, pred_result], axis=0).reset_index(drop=True)

        if "P510" in list(pred_result['code']):
            P510 =  pred_result[pred_result['code'].isin(['P510'])]
            pred_result = pred_result[~pred_result['code'].isin(['P510'])]
            P510 = P510[P510['confidence'] > 0.3]
            pred_result = pd.concat([P510, pred_result], axis=0).reset_index(drop=True)

        if "P306" in list(pred_result['code']):
            P306 =  pred_result[pred_result['code'].isin(['P306'])]
            pred_result = pred_result[~pred_result['code'].isin(['P306'])]
            P306 = P306[P306['confidence'] > 0.15]
            pred_result = pd.concat([P306, pred_result], axis=0).reset_index(drop=True)

        if "P302" in list(pred_result['code']):
            P302 =  pred_result[pred_result['code'].isin(['P302'])]
            pred_result = pred_result[~pred_result['code'].isin(['P302'])]
            P302 = P302[P302['confidence'] > 0.2]
            pred_result = pd.concat([P302, pred_result], axis=0).reset_index(drop=True)

        if "P304" in list(pred_result['code']):
            P304 =  pred_result[pred_result['code'].isin(['P304'])]
            pred_result = pred_result[~pred_result['code'].isin(['P304'])]
            P304 = P304[P304['confidence'] > 0.15]
            pred_result = pd.concat([P304, pred_result], axis=0).reset_index(drop=True)

        if "P304_re" in list(pred_result['code']):
            P304_re =  pred_result[pred_result['code'].isin(['P304_re'])]
            pred_result = pred_result[~pred_result['code'].isin(['P304_re'])]
            P304_re = P304_re[P304_re['confidence'] > 0.15]
            pred_result = pd.concat([P304_re, pred_result], axis=0).reset_index(drop=True)
        return pred_result


    def infer(self, img_json):
        defect = 0
        status = 200
        imgsz = self.paramers['imgsz']
        ok_code = self.paramers['ok_code']
        ok_conf = self.paramers['ok_conf']
        conf_thres = self.paramers['conf_thres']
        iou_thres = self.paramers['iou_thres']
        thresholds = self.paramers["code_threshold"]
        priority = self.paramers["priority"]

        saveROOT_PATH = img_json['info']['saveROOT_PATH']
        img_uid = img_json['image'][0]['uid']
        img_type = img_json['image'][0]['type']
        img_gid = img_json['image'][0]['gid']
        test_img = img_json['image'][0]["path"]

        assert os.path.exists(test_img)
        img = cv2.imread(test_img)

        pred = self.data_proecess(self.model, img, imgsz, conf_thres, iou_thres)
        pred_result = pd.DataFrame(pred, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'code'])
        pred_result['code'] = pred_result['code'].astype(int)
        pred_result['code'] = pred_result['code'].map(lambda x: self.classes[x])

        pred_result = self.pred_filter(pred_result)

        if pred_result.shape[0] == 0:
            pred_result.loc[0] = [0, 0, 0, 0, ok_conf, ok_code]
            final = pred_result.loc[0]
            self.result = self.creat_dict()
            self.dump_result(self.result, pred_result, final, status, img_uid, img_gid, defect, img_type, saveROOT_PATH)
            return self.result

        else:
            defect = pred_result.shape[0]
            final = self.get_final_code(pred_result, priority, thresholds)
            self.result = self.creat_dict()
            self.dump_result(self.result, pred_result, final, status, img_uid, img_gid, defect, img_type, saveROOT_PATH)
            return self.result
