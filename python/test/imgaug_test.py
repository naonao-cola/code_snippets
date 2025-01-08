import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import json
import cv2
import os
import random
import glob
from pathlib import Path


def augment_data(image_path, annotation_path, output_dir, index):
    image_base_name = os.path.basename(image_path).split('.')[0]
    # 读取图像和标注文件
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)

    # 将labelme标注数据转换为imgaug的BoundingBoxesOnImage格式
    bbs = []
    for shape in annotation_data['shapes']:
        label = shape['label']
        points = shape['points']
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]
        bb = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)
        bbs.append(bb)
    bbs_on_image = BoundingBoxesOnImage(bbs, shape=image.shape)

    # 定义图像增强器
    seq = iaa.Sequential([
        iaa.Multiply(mul=(0.8, 1.2)),
        iaa.Sometimes(0.2, iaa.AddToHueAndSaturation(
            value=(-10, 10), per_channel=True)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1, 3.0))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(
            scale=(0, random.random() * 0.05 * 255))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.25))),
        iaa.Multiply((0.9, 1.1), per_channel=0.2),
        iaa.Fliplr(0.1),  # 水平翻转
        # iaa.Affine(scale=random.uniform(0.9, 1.1)),
        # iaa.Affine(rotate=(-5, 5)),
        # iaa.PerspectiveTransform(scale=(0, 0.05)),
    ])

    # 进行图像增强
    augmented_image, augmented_bbs_on_image = seq(
        image=image, bounding_boxes=bbs_on_image)

    # 生成labelme标注
    transformed_annotation = {
        'version': '5.3.1',
        'flags': {},
        'shapes': [],
        'imagePath': f'{image_base_name}_aug_{index}.jpg',
        'imageData': None,
        'imageHeight': augmented_image.shape[0],
        'imageWidth': augmented_image.shape[1]
    }

    for bb in augmented_bbs_on_image.bounding_boxes:
        transformed_shape = {
            'label': bb.label,
            'points': [[float(bb.x1), float(bb.y1)], [float(bb.x2), float(bb.y2)]],
            'group_id': None,
            'description': '',
            'shape_type': 'rectangle',
            'flags': shape['flags']
        }
        transformed_annotation['shapes'].append(transformed_shape)

    # 导出增强后的图像和标注
    output_image_path = os.path.join(
        output_dir, f'{image_base_name}_aug_{index}.jpg')
    output_annotation_path = os.path.join(
        output_dir, f'{image_base_name}_aug_{index}.json')

    cv2.imwrite(output_image_path, augmented_image)
    with open(output_annotation_path, 'w') as f:
        json.dump(transformed_annotation, f)


# for i in range(30):
#     augment_data('./data/Right_20240430151354405.jpg',
#                  './data/Right_20240430151354405.json', './aug2', i)


def apply_aug(img_path, dst_path):
    search_path = img_path + '/*.JPG'
    file_list = glob.glob(search_path, recursive=True)
    for file in file_list:
        img_path = file
        json_path = Path(file).with_suffix('.json')
        if not json_path.exists():
            continue
        print(json_path)
        with open(json_path, 'r')as f:
            json_data = json.load(f)
        index = 5
        # 对黑色海绵 地线巴做多增强
        # for i, label in enumerate(json_data['shapes']):
        #     if json_data['shapes'][i]['label'] == "DXB" or json_data['shapes'][i]['label'] == "AXFSJ":
        #         index = 8
        #         break

        for i in range(index):
            augment_data(img_path, json_path, dst_path, i)


apply_aug(r"Y:\proj\www\repo\yolo8_test\dataset\sbg_hl_test\result",
          r'Y:\proj\www\repo\yolo8_test\dataset\sbg_hl_test\result\aug')
