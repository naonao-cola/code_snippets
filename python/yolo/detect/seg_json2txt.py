import os
import json
from pathlib import Path
import glob


import cv2
import numpy as np
import random

labels = ["lr", "in", "top"]


def seg_json2txt(json_path, save_path):
    file = glob.glob(json_path + '/**/*.' + "json", recursive=True)
    for file_item in file:
        path_json = file_item
        path_txt = save_path + "/" + Path(file_item).stem + ".txt"

        with open(path_json, 'r', encoding='utf-8') as path_json:
            jsonx = json.load(path_json)
        img_w = jsonx["info"]["width"]
        img_h = jsonx["info"]["height"]
        objects = jsonx['objects']
        with open(path_txt, 'w+') as ftxt:
            for object in objects:
                lable = object["category"]
                label_index = labels.index(lable)
                points = object["segmentation"]
                points_nor_list = []
                for pt in points:
                    points_nor_list.append(pt[0]/img_w)
                    points_nor_list.append(pt[1]/img_h)
                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_list = " ".join(points_nor_list)
                label_str = str(label_index) + " " + points_nor_list + "\n"
                ftxt.write(label_str)


def seg_txt2json(txt_path, save_path, img_width=5472, img_height=3648):
    files = glob.glob(txt_path + '/**/*.' + "txt", recursive=True)

    for txt_item in files:
        lines = []
        with open(txt_item, 'r') as file:
            for line in file:
                lines.append(line.strip().split())

        transformed_annotation = {}
        transformed_annotation["info"] = {
            "description": "ISAT",
            "folder": txt_path,
            "name": Path(txt_item).stem + ".jpg",
            "width": img_width,
            "height": img_height,
            "depth": 3,
            "note": ""
        }
        transformed_annotation["objects"] = []
        # 循环行
        for idx, line_item in enumerate(lines):
            object_item = {}
            object_item["category"] = labels[int(line_item[0])]
            object_item["group"] = idx+1
            object_item["segmentation"] = []
            object_item["area"] = 0.0
            object_item["layer"] = 1.0
            object_item["bbox"] = []
            object_item["iscrowd"] = "false"
            object_item["note"] = ""
            # 循环每一行
            x, y = 0.0, 0.0
            for idx_pt, pt in enumerate(line_item[1:]):
                if idx_pt % 2 == 0:
                    x = float(pt) * img_width
                else:
                    y = float(pt) * img_height
                    object_item["segmentation"].append([x, y])
            transformed_annotation["objects"].append(object_item)

        output_annotation_path = Path(txt_item).with_suffix('.json')
        with open(output_annotation_path, 'w') as f:
            json.dump(transformed_annotation, f)


# seg_json2txt(r"E:\demo\pyt\data", r"E:\demo\pyt\data")


# seg_txt2json(r"E:\demo\pyt\data", r"E:\demo\pyt\data\1",
#              img_width=5472, img_height=3648)


def seg_json2mask(json_path, save_path):
    file = glob.glob(json_path + '/**/*.' + "json", recursive=True)
    for file_item in file:
        path_json = file_item
        path_txt = save_path + "/" + Path(file_item).stem + ".jpg"

        colors = []

        for label in range(len(labels)):
            # 生成随机颜色
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            colors.append(color)

        with open(path_json, 'r', encoding='utf-8') as path_json:
            jsonx = json.load(path_json)
        img_w = jsonx["info"]["width"]
        img_h = jsonx["info"]["height"]
        objects = jsonx['objects']
        mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        for idx, object in enumerate(objects):
            points = object["segmentation"]
            points_nor_list = []
            lable = object["category"]
            label_index = labels.index(lable)
            for pt in points:
                points_nor_list.append([pt[0], pt[1]])

            pt_vec = np.array(points_nor_list, dtype=np.int32)
            cv2.fillConvexPoly(mask, pt_vec, colors[label_index])
        cv2.imwrite(path_txt, mask)


seg_json2mask(r"E:\demo\pyt\data", r"E:\demo\pyt\data")
