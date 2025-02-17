import os
import json
from pathlib import Path
import glob
import yaml
from tqdm import tqdm
import cv2
import os

# 金针需要裁图
left_x = 1272
left_y = 416
temp_width = 2256
temp_height = 2072


class data_parse:
    def __init__(self, yaml_txt):
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
        self.colors = colors
        self.labels = labels

    def seg_json2txt(self, json_path, save_path):
        file = glob.glob(json_path + '/**/*.' + "json", recursive=True)
        for file_item in tqdm(file):
            path_json = file_item
            path_txt = save_path + "/" + Path(file_item).stem + ".txt"

            with open(path_json, 'r', encoding='utf-8') as path_json:
                jsonx = json.load(path_json)
            # img_w = jsonx["info"]["width"]
            # img_h = jsonx["info"]["height"]
            img_w = temp_width
            img_h = temp_height

            objects = jsonx['objects']
            with open(path_txt, 'w+') as ftxt:
                for object in objects:
                    lable = object["category"]
                    label_index = self.labels.index(lable) - 1
                    points = object["segmentation"]
                    points_nor_list = []
                    for pt in points:
                        points_nor_list.append((pt[0] - left_x)/img_w)
                        points_nor_list.append((pt[1] - left_y)/img_h)
                    points_nor_list = list(
                        map(lambda x: str(x), points_nor_list))
                    points_nor_list = " ".join(points_nor_list)
                    label_str = str(label_index) + " " + points_nor_list + "\n"
                    ftxt.write(label_str)

    def seg_txt2json(self, txt_path, save_path, img_width=5472, img_height=3648):
        files = glob.glob(txt_path + '/**/*.' + "txt", recursive=True)

        for txt_item in tqdm(files):
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
                object_item["category"] = self.labels[int(line_item[0])] + 1
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

    def crop_img(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp','.PNG'))]
        for image_file in image_files:
            input_path = os.path.join(img_path, image_file)
            img = cv2.imread(input_path)
            if img is None:
                print(f"Error: Unable to read {input_path}")
                continue
            x1, y1, x2, y2 = left_x, left_y, left_x + temp_width, left_y + temp_height
            cropped_img = img[y1:y2, x1:x2]
            output_path = os.path.join(save_path, image_file)
            # 保存裁剪后的图片
            cv2.imwrite(output_path, cropped_img)
            print(f"Processed {input_path} and saved to {output_path}")


if __name__ == "__main__":


    # 金针
    yaml_txt = r"/home/tvt/luobing/hf_dataset/jinzhen_seg/origin_img/isat.yaml"
    m_tool = data_parse(yaml_txt)
    #m_tool.crop_img("/home/tvt/luobing/hf_dataset/jinzhen_seg/origin_img/", "/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/seg/")

    json_path = r"/home/tvt/luobing/hf_dataset/jinzhen_seg/origin_img/"
    save_path = r"/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/label/"
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    m_tool.seg_json2txt(json_path, save_path)