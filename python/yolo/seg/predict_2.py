import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from detials import *
import torch

# Load a model
#model = YOLO('/data/lb/hf_dataset/v1/out16/weights/best.pt', task="segment")
# Run batched inference on a list of images
#results = model.predict("/data/lb/hf_dataset/1119_template/gray_far/11.jpg",
#                        device=0)

#for result in results:
#    if result.masks is not None:
#        boxes = result.boxes
#        masks = result.masks   # 获取分割mask
#       masks = masks.data     # 获取mask数据
#        boxes_cls = boxes.cls  # 获取mask对应的类别ID
#        class_ids = result.masks.cls  # 获取mask对应的类别ID
#        mask_raw = masks[0].cpu().numpy().astype(np.uint8) * 255
#        for i, (mask, class_id) in enumerate(zip(masks, boxes_cls)):
#            mask_raw += mask.cpu().numpy().astype(np.uint8) * 255
#        dst = cv2.resize(mask_raw, (914, 324))
#        cv2.imwrite(f'./1.png', dst)


model = YOLO('/data/lb/hf_dataset/v1/out16/weights/best.pt', task="segment")


def predict_segmentation(img_path, template_path, model_path=""):
    src_img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
    temp_img = cv2.imread(template_path, cv2.COLOR_BGR2GRAY)
    box_list = match_template(src_img, temp_img, threshold=0.6)
    p = Path(img_path)
    transformed_annotation = {}
    transformed_annotation["info"] = {
        "description": "ISAT",
        "folder": str(p.parent),
        "name": p.name,
        "width": src_img.shape[1],
        "height": src_img.shape[0],
        "depth": 3,
        "note": ""
    }
    transformed_annotation["objects"] = []
    category = ["top", "lr", "in"]
    count = 1
    for index, bbox in enumerate(box_list):
        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[2], bbox[3])
        crop_img = src_img[top_left[1]:bottom_right[1],
                           top_left[0]:bottom_right[0]]
        results = model.predict(crop_img, device=0, retina_masks=True)
        #results = model.predict(crop_img, device=0)

        for result in results:
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks   # 获取分割mask
                masks = masks.data     # 获取mask数据
                boxes_cls = boxes.cls  # 获取mask对应的类别ID
                # print("boxes_cls ", boxes_cls)
                for i, (mask, class_id) in enumerate(zip(masks, boxes_cls)):
                    mask_raw = mask.cpu().numpy().astype(np.uint8) * 255
                    dst = cv2.resize(
                        mask_raw, (crop_img.shape[1], crop_img.shape[0]))
                    contours, hierarchy = cv2.findContours(
                        dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        x += top_left[0]
                        y += top_left[1]
                        polygon = [
                            [int(point[0][0] + top_left[0]), int(point[0][1] + top_left[1])] for point in contour]
                        # print("category ", class_id)
                        object_item = {}
                        object_item["category"] = category[torch.floor(
                            class_id).to(torch.int)]
                        object_item["group"] = count
                        object_item["segmentation"] = polygon
                        object_item["area"] = 0.0
                        object_item["layer"] = 1.0
                        object_item["bbox"] = [
                            int(x), int(y), int(x+w), int(y+h)]
                        object_item["iscrowd"] = "false"
                        object_item["note"] = ""
                        count += 1
                        transformed_annotation["objects"].append(object_item)
    save_annotation_path = p.with_suffix('.json')
    with open(save_annotation_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_annotation, f)


 predict_segmentation("/data/lb/hf_dataset/1118/6fe7e5c974754aa0_0_20241120143514086.PNG",
                      "/data/lb/hf_dataset/1119_template/gray_far/11.jpg")

#file_list = ["black", "gray_far", "gray_near", "other"]
#for floder in file_list:
#     img_root = "/data/lb/hf_dataset/1120/" + floder
#     print("floder:", floder)
#     template_root = "/data/lb/hf_dataset/1119_template"
#     template_floder = f"{template_root}/{floder}"
#     for temp in os.listdir(template_floder):
#         templateImg_path = f"{template_floder}/{temp}"
#
#         for file in os.listdir(img_root):
#             img_path = f"{img_root}/{file}"
#             predict_segmentation(img_path, templateImg_path)

