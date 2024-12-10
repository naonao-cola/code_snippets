import cv2
import numpy as np
import glob
from tqdm import tqdm
import os
import json
import shutil
import yaml
from pathlib import Path



def init(yaml_txt):
    # 读取 YAML 文件
    with open(yaml_txt, 'r') as file:
        data = yaml.safe_load(file)
    # 访问 'label' 部分
    info = data['label']
    # 获取所有标签信息
    colors = []
    labels = []
    for label in info:
        hex_color = label['color']
        name = label['name']

        labels.append(name)
        # 解析颜色
        # 去掉 `#` 符号
        hex_color = hex_color.lstrip('#')
        # 解析十六进制颜色
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        colors.append([b, g, r])

        # print(f"Name: {name}, Color: {color}")
    colors = colors
    labels = labels

def calc_isIn(box_big, box_samll):
    x1, y1, x2, y2 = box_big
    x1p, y1p, x2p, y2p = box_samll
    if x1p >= x1 and y1p >= y1 and x2p <= x2 and y2p <= y2:
        return True
    else:
        return False

def compute_iou(box1, box2):

    # 计算两个框的交集区域
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    # 计算交集区域的坐标
    ix1 = max(x1, x1p)
    iy1 = max(y1, y1p)
    ix2 = min(x2, x2p)
    iy2 = min(y2, y2p)

    # 计算交集区域的面积
    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)
    intersection = iw * ih

    # 计算两个框的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2p - x1p) * (y2p - y1p)

    # 计算并集区域的面积
    union = area1 + area2 - intersection

    # 计算IOU
    iou = intersection / union
    return iou

def convert_box(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [x, y, x + w, y + h]

def convert_box_inv(box):
    x = box[0]
    y = box[1]
    w = box[2] - x
    h = box[3] - y
    return [x, y, w, h]

def filter_boxes(box_one, box_list, iou_threshold=0.5):
    flag = True
    for box in box_list:
        iou = compute_iou(box_one, box)
        if iou > iou_threshold:
            flag = False

    if flag:
        box_list.append(box_one)
    return box_list

def match_template(source_img, template_img, threshold=0.6):
    if len(source_img.shape) == 3:
        src_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        src_img = source_img

    if len(template_img.shape) == 3:
        tmp_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    else:
        tmp_img = template_img

    # 执行模板匹配
    result = cv2.matchTemplate(src_img, tmp_img, cv2.TM_CCOEFF_NORMED)
    h, w = tmp_img.shape
    locations = np.where(result >= threshold)

    box_list = []
    for pt in zip(*locations[::-1]):
        top_left = pt
        bottom_right = (top_left[0] + w, top_left[1] + h)
        if len(box_list) < 1:
            box_list.append([top_left[0], top_left[1],
                            bottom_right[0], bottom_right[1]])
        if len(box_list) >= 1:
            box_one = [top_left[0], top_left[1],
                        bottom_right[0], bottom_right[1]]
            box_list = filter_boxes(box_one, box_list)

    return box_list

def crop_json(self, crop_box, json_path, save_path):
    # 新小图标注文件
    w = int(crop_box[2] - crop_box[0])
    h = int(crop_box[3] - crop_box[1])
    transformed_annotation = {}
    name_jpg = os.path.basename(save_path)
    name = os.path.splitext(name_jpg)[0]
    folder_path = save_path.split(name)[0]

    transformed_annotation["info"] = {
        "description": "ISAT",
        "folder": folder_path,
        "name": name_jpg,
        "width": w,
        "height": h,
        "depth": 3,
        "note": ""
    }
    transformed_annotation["objects"] = []
    # 获取原标注信息
    with open(json_path, 'r', encoding='utf-8') as gt_file:
        json_info = json.load(gt_file)
    img_w = json_info["info"]["width"]
    img_h = json_info["info"]["height"]
    objects = json_info['objects']

    # 存储整张图的标注信息
    segments_list = []
    cate_list = []
    bbox_list = []

    for object in objects:
        cate_list.append(object["category"])
        segments_list.append(object["segmentation"])
        bbox_list.append(object["bbox"])

    # 判断属于该小图的标注
    count = 1
    for i in range(len(bbox_list)):
        rst = self.calc_isIn(crop_box, bbox_list[i])
        if rst:
            object_item = {}
            object_item["category"] = cate_list[i]
            object_item["group"] = count
            object_item["segmentation"] = []
            object_item["area"] = 0.0
            object_item["layer"] = 1.0
            object_item["bbox"] = bbox_list[i]
            object_item["iscrowd"] = "false"
            object_item["note"] = ""
            count += 1
            for pnt in segments_list[i]:
                object_item["segmentation"].append(
                    [pnt[0]-crop_box[0], pnt[1]-crop_box[1]])
            transformed_annotation["objects"].append(object_item)

    save_annotation_path = f"{folder_path}\\{name}.json"
    with open(save_annotation_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_annotation, f, indent=4)




