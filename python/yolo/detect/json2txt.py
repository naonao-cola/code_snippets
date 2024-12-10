import os
import json
from pathlib import Path
sets = ['train', 'test', 'val']


# classes = ["tb",
#            "ibj",
#            "marly",
#            "wa_gray_s",
#            "wa_black_s",
#            "screw_head",
#            "leak_hole",
#            "bs",
#            "gwb",
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


# classes = ["CCC",
#            "ASTA",
#            "CE",
#            "KEMA",
#            "UL",
#            ]
# classes = ["T",
#           "pushoff",
#           "legend",
#           "counter",
#           "closing_white",
#           "screw",
#           "closing_door",
#           "evopact",
#           "schneider",
#           "pushon",
#           "rect",
#           "closing_green",
#           "battery",
#           "battery_screw",
#           "T1",
#           "T2",
#           "environmental",
#           "dangerous",
#           "model",
#           "hanging_board",
#           "hole_plug",
#           "drive_plate",
#           "drive_label",
#           "closing_label",
#           "T3"
#           ]
# classes = ["anniu_hong",
#            "anniu_lv",
#            "biaoqian",
#            "jishuqi",
#            "hefen_fir",
#            "hefen_sec",
#            "hefen_thr",
#            "hefen_fou",
#            "screw_black",
#            "hefen_door",
#            "guagoubiaoqian",
#            "hang_board",
#            "holeplug",
#            "screw",
#            "grounding"
#            ]

# classes = ["guagou",
#           "hole_plug",
#           "hanging_board",
#           "ground_label",
#           "ip_screw",
#           "button_green",
#           "button_red",
#           "counter",
#           "model",
#           "closing_door",
#           "screw_black",
#           "hefen",
#           "chuneng",
#           "diandong_label",
#           "smart_label",
#           "schneider"
#           ]
#
#


classes = ["T",
          "pushoff",
          "legend",
          "counter",
          "closing_white",
          "screw",
          "closing_door",
          "evopact",
          "schneider",
          "pushon",
          "rect",
          "closing_green",
          "closing_red",
          "battery",
          "battery_screw",
          "T1",
          "T2",
          "environmental",
          "dangerous",
          "model",
          "hanging_board",
          "hole_plug",
          "drive_plate",
          "drive_label",
          "closing_label",
          "T3"
          ]


def convert_annotation(image_id):

    path_json = '/data/proj/www/repo/yolo8_test/dataset/digita/Annotations/%s.json' % (
        image_id)
    path_txt = '/data/proj/www/repo/yolo8_test/dataset/digita/labels/%s.txt' % (
        image_id)

    with open(path_json, 'r', encoding='utf-8') as path_json:
        jsonx = json.load(path_json)
        with open(path_txt, 'w+') as ftxt:
            shapes = jsonx['shapes']
            # 获取图片长和宽
            width = jsonx['imageWidth']
            height = jsonx['imageHeight']
            for shape in shapes:
               # 获取矩形框两个角点坐标
                x1 = shape['points'][0][0]
                y1 = shape['points'][0][1]
                x2 = shape['points'][1][0]
                y2 = shape['points'][1][1]

                cat = shape['label']

                cls_id = classes.index(cat)
                # 对结果进行归一化
                dw = 1. / width
                dh = 1. / height
                x = dw * (x1+x2)/2
                y = dh * (y1+y2)/2
                w = dw * abs(x2-x1)
                h = dh * abs(y2 - y1)
                yolo = f"{cls_id} {x} {y} {w} {h} \n"
                ftxt.writelines(yolo)


def mk_dataset():
    for image_set in sets:
        if not os.path.exists("/data/proj/www/repo/yolo8_test/dataset/digita/labels/"):
            os.mkdirs("/data/proj/www/repo/yolo8_test/dataset/digita/labels/")
        image_ids = open('/data/proj/www/repo/yolo8_test/dataset/digita/ImageSets/%s.txt' %
                         (image_set)).read().strip().split()
        list_file = open('/data/proj/www/repo/yolo8_test/dataset/digita/%s.txt' %
                         (image_set), 'w')
        for image_id in image_ids:
            file_name = '/data/proj/www/repo/yolo8_test/dataset/digita/Annotations/%s.json' % (
                image_id)
            img_file = Path(file_name)
            if not img_file.exists():
                print(img_file)
                continue
            list_file.write(
                '/data/proj/www/repo/yolo8_test/dataset/digita/images/%s.jpg\n' % (image_id))
            # 调用  year = 年份  image_id = 对应的文件名_id
            convert_annotation(image_id)
        # 关闭文件
        list_file.close()


mk_dataset()
