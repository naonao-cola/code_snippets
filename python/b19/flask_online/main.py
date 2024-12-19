# {PyArmor Plugins}

import os
import json
from predictor import Model
import pandas as pd
from tqdm import tqdm
import datetime
import shutil
import cv2
import random
from tool.util_tool import creat_xml

# import openpyxl

def init(path):
    # check_files(path)

    model__json = {}
    model_path = os.path.join(path, 'conf/best.pt')
    model_param = read_json(os.path.join(path, 'conf/hyper_param.json'))

    model__json['data'] = 'conf/coco.ymal'
    model__json['model'] = model_path
    model__json['paramers'] = model_param

    # predictor = Model(color_spec, common_spec=common_spec)
    predictor = Model(model__json)

    return predictor


def predict(predictor, img_json):
    time1 = datetime.datetime.now()
    print('\n', time1, '---mask', "---", img_json)
    result_json = predictor.infer(img_json)
    time6 = datetime.datetime.now()
    print(time6, 'end', '\n')
    return result_json


def check_files(path):
    files = ['color', 'hyper_param.json']
    for file in files:
        assert file in os.listdir(path), '{} not in package!'.format(file)


def read_txt(file):
    with open(file, 'r') as f:
        content = f.read().splitlines()
    return content


def read_json(file):
    with open(file, 'r') as f:
        content = json.load(f)
    return content


def draw_bbox(bbox, img, label, score, num, line_thickness=3):
    img = cv2.imread(img)
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # print("tl：{}" .format(tl))
    color = [random.randint(0, 255) for _ in range(3)]
    color = tuple(color)
    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))  # 左上角和右下角
    offset = int(bbox[1]) + num * 24
    c3, c4 = (int(bbox[2]), int(bbox[1])), (int(bbox[2]), offset)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # cv2.line(img, c3, c4, color, thickness=1, lineType=cv2.LINE_AA)
    tf = max(tl - 2, 1)
    score = round(score, 3)  # 四舍五入，保留3位小数
    cv2.putText(img, label + ":" + str(score), (c4[0], c4[1] - 2), 0, tl / 5, [255, 255, 255], thickness=tf,
                lineType=cv2.LINE_AA)
    return img

if __name__ == "__main__":
    # PyArmor Plugin: check_multi_mac()
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    path = "./"
    save_dir = '/data/hjx/B19/data/out'
    test_path = '/data/hjx/B19/data/BeforeTest'

    predictor = init(path)

    data = []
    for root, _, files in os.walk(test_path):
        for file in files:
            if file.split('.')[-1] in ['JPG', 'jpg']:
                img_path = os.path.join(root, file)
    # img_list = [r"../Images/495.jpg"]
    # gls_id = os.path.basename(img_path).split("_")[0]
                img_json = {
                    "image": [
                        {
                        "path": img_path,#本图有code
                        "type": "Color",
                        "uid": "imageid1",
                        "gid": "groupid"},
                    ],
                    "model": {
                        "type": "AOI"},
                    "info": {
                        "PRODUCT_ID": "P647F1FEB100",
                        "UNIT_ID": "AOI800",
                        "GLASS_ID": "W28P70302",
                        "model_dir": "./",
                        "saveROOT_PATH": "/data/hjx/hehui/data/out"}
                }
                try:
                    result = predictor.infer(img_json)
                    gt = root.split('/')[-1]
                    preds = result['result'][0]['img_cls']
                    scores = result['result'][0]['img_score']
                    boxes = result['result'][0]['img_box']
                    data.append({"gt": gt, "pred": preds[0], "score": float(scores[0]), "file": file})
                    print(result)

                    if gt != preds[0]:
                        obj = os.path.join(save_dir, gt, preds[0])
                        if not os.path.exists(obj):
                            os.makedirs(obj)
                        shutil.copy(img_path, obj)
                    if gt == preds[0]:
                        obj = os.path.join(save_dir, gt, gt)
                        if not os.path.exists(obj):
                            os.makedirs(obj)
                        shutil.copy(img_path,obj)
                        # shutil.move(os.path.join(save_dir, file), obj)
                    df = pd.DataFrame(data)
                    df.to_excel('/data/hjx/B19/data/out/result.xlsx')


                    # 画框
                    # for i, pred in enumerate(preds):
                    #     bbox = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][-1]]
                    #     draw_result = draw_bbox(bbox, os.path.join(obj, file), pred, scores[i], 0)
                    #     cv2.imwrite(os.path.join(obj, file), draw_result)

                    xml_path = '/data/hjx/hehui/code/A1AOIH020.DL_770.143H1P77.A13A0201.A13A020101.0181.b.1698064087.xml'
                    creat_xml(boxes, preds, os.path.join(obj, file), xml_path, '1024', '768')




                except:
                    pass

