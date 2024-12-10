from ultralytics import YOLO
from pathlib import Path
from pathlib import Path
from PIL import Image
import glob
import json

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/digita/default.yaml"
DATA_YAML  = "/data/proj/www/repo/yolo8_test/cfg/digita/lego.yaml"
MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/digita/yolov8l.yaml"

# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_hl/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_hl/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_hl/yolov8l.yaml"

# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_il/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_il/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_il/yolov8l.yaml"


# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/yolov8l.yaml"

# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_biao/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_biao/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/sbg_biao/yolov8l.yaml"

# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_o2_150/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_o2_150/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_o2_150/yolov8l.yaml"


# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_k_210/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_k_210/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/evopact_k_210/yolov8l.yaml"


# TRAIN_YAML = "/data/proj/www/repo/yolo8_test/cfg/default.yaml"
# DATA_YAML = "/data/proj/www/repo/yolo8_test/cfg/lego.yaml"
# MODEL_YAML = "/data/proj/www/repo/yolo8_test/cfg/yolov8n.yaml"


def train_obj(device=1):
    model = YOLO(Path(MODEL_YAML))
    model.train(data=DATA_YAML, cfg=TRAIN_YAML, device=device)
    model.val()
    model.export(format='onnx')  # 将模型导出为 ONNX 格式


def predit_obj(img):
    model = YOLO(
        Path("/data/proj/www/repo/yolo8_test/build/sbg_il/out/weights/best.pt"), task="detect")
    ret = model.predict(
        source=img)  # 对图像进行预测
    for item in ret:
        boxes = item.boxes
        masks = item.masks
        keypoints = item.keypoints
        probs = item.probs
        print("boxes: ", boxes, "\n")
        print("mask: ", masks, "\n")
        print("keypoints: ", keypoints, "\n")
        print("probs: ", probs, "\n")

        im_array = item.plot(conf=False, line_width=1, font_size=1.5)
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        im.save('results.jpg')  # save image
        return im


def predict_yolo8(img_path, save_path):
    search_path = img_path + '/**/*.*'
    file_list = glob.glob(search_path, recursive=True)
    model = YOLO(Path(
        "/data/proj/www/repo/yolo8_test/build/lego/out/weights/best.pt"), task="detect")
    # results = model(file_list, conf=0.6)  # return a list of Results objects
    for img_path in file_list:
        ret = model.predict(img_path, conf=0.5, iou=0.5)
        for item in ret:
            boxes = item.boxes
            print(boxes.cls)
            print(boxes.xyxy)
            # masks = item.masks
            # keypoints = item.keypoints
            # probs = item.probs
            im_array = item.plot(conf=False, line_width=4, font_size=5)
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(save_path + Path(img_path).name)  # save image
            item.save_txt(save_path + Path(img_path).stem + ".txt")


def txt2json(txt_path, label_vec):
    search_path = txt_path + '*.txt'
    file_list = glob.glob(search_path, recursive=True)
    img_width = 4032
    img_height = 3024
    # 读取每一个文本文件
    for txt_item in file_list:
        lines = []
        with open(txt_item, 'r') as file:
            for line in file:
                lines.append(line.strip().split())
        transformed_annotation = {
            'version': '5.3.1',
            'flags': {},
            'shapes': [],
            'imagePath': Path(txt_item).stem + ".jpg",
            'imageData': None,
            'imageHeight': img_height,
            'imageWidth': img_width
        }
        # 处理文件
        for line_item in lines:
            print("current line: ", line_item)
            line_label = label_vec[int(line_item[0])]
            # 坐标是中心点左边
            dx = float(line_item[1])
            dy = float(line_item[2])
            dw = float(line_item[3])
            dh = float(line_item[4])
            x = dx * img_width
            y = dy * img_height
            w = dw * img_width
            h = dh * img_height
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            transformed_shape = {
                'label': line_label,
                'points': [[float(x1), float(y1)], [float(x2), float(y2)]],
                'group_id': None,
                'description': '',
                'shape_type': 'rectangle',
                'flags': {}
            }
            transformed_annotation['shapes'].append(transformed_shape)
        output_annotation_path = Path(txt_item).with_suffix('.json')
        with open(output_annotation_path, 'w') as f:
            json.dump(transformed_annotation, f)


# predict_yolo8(
#     "/data/proj/www/repo/yolo8_test/dataset/digital_test/data/", "/data/proj/www/repo/yolo8_test/dataset/digital_test/result/")

# label_vec = ["tb",
#              "ibj",
#              "marly",
#              "wa_gray_s",
#              "wa_black_s",
#              "screw_head",
#              "leak_hole",
#              "bs",
#              "gwb",
#              ]

# classes = ["CCC",
#            "ASTA",
#            "CE",
#            "KEMA",
#            "UL",
#            ]

# classes = ["KD",
#            "HJ",
#            "TB",
#            "DXB",
#            "FSJ",
#            "AXFSJ",
#            "ML",
#            "HSFHJ",
#            "LS",
#            "BOX",
#            ]

# classes = ["T",
#            "pushoff",
#            "legend",
#            "counter",
#            "closing_white",
#            "screw",
#            "closing_door",
#            "evopact",
#            "schneider",
#            "pushon",
#            "rect",
#            "closing_green",
#            "battery",
#            "battery_screw",
#            "T1",
#            "T2",
#            "environmental",
#            "dangerous",
#            "model",
#            "hanging_board",
#            "hole_plug",
#            "drive_plate",
#            "drive_label",
#            "closing_label",
#            "T3"
#            ]

# txt2json("/data/proj/www/repo/yolo8_test/dataset/digital_test/result/", classes)

train_obj(device=1)
