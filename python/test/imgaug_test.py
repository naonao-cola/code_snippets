import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import json
import cv2
import os
import random
import glob
from pathlib import Path


# 修改bug  https://github.com/aleju/imgaug/issues/859

def parse_labelme(json_path, img_shape):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bbs, kps, polys, shape_meta = [], [], [], []
    h, w = img_shape[:2]

    for shape in data["shapes"]:
        label = shape["label"]
        pts = shape["points"]

        if shape["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = pts
            bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
            shape_meta.append(shape)

        elif shape["shape_type"] == "point":
            x, y = pts[0]
            kps.append(Keypoint(x=x, y=y))
            shape_meta.append(shape)

        elif shape["shape_type"] == "polygon":
            # 转成 imgaug.Polygon
            polys.append(Polygon([Keypoint(x=p[0], y=p[1]) for p in pts]))
            shape_meta.append(shape)

    bbs_on = BoundingBoxesOnImage(bbs, shape=img_shape) if bbs else None
    kps_on = KeypointsOnImage(kps, shape=img_shape) if kps else None
    polys_on = PolygonsOnImage(polys, shape=img_shape) if polys else None

    return bbs_on, kps_on, polys_on, shape_meta, data


# ---------- 2. 写回 labelme ----------
def rebuild_labelme(aug_img, bbs_aug, kps_aug, polys_aug, shape_meta, imagePath):
    h, w = aug_img.shape[:2]
    new_shapes = []

    bb_idx = kp_idx = poly_idx = 0
    for shape in shape_meta:
        label = shape["label"]
        new_shape = shape.copy()

        if shape["shape_type"] == "rectangle":
            bb = bbs_aug.bounding_boxes[bb_idx]
            bb_idx += 1
            new_shape["points"] = [[float(bb.x1), float(bb.y1)], [float(bb.x2), float(bb.y2)]]

        elif shape["shape_type"] == "point":
            kp = kps_aug.keypoints[kp_idx]
            kp_idx += 1
            new_shape["points"] = [[float(kp.x), float(kp.y)]]

        elif shape["shape_type"] == "polygon":
            poly = polys_aug.polygons[poly_idx]
            poly_idx += 1
            # 取回增强后的顶点
            new_shape['points'] = poly.exterior.astype(float).tolist()

        new_shapes.append(new_shape)

    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": new_shapes,
        "imagePath": imagePath,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def augment_data(img_path, json_path, out_dir, idx):
    img = cv2.imread(img_path)
    bbs, kps, polys, shape_meta, json_data = parse_labelme(json_path, img.shape)

    seq = iaa.Sequential([
        iaa.Multiply((0.8, 1.2)),
        iaa.Sometimes(0.2, iaa.AddToHueAndSaturation((-10, 10), per_channel=True)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(1, 3))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.25))),
        iaa.Multiply((0.9, 1.1), per_channel=0.2),
        iaa.Fliplr(0.1)
    ])

    aug_img, aug_bbs, aug_kps, aug_polys = seq(image=img, bounding_boxes=bbs, keypoints=kps, polygons=polys)

    if aug_bbs is not None:
        aug_bbs = aug_bbs.clip_out_of_image()
    if aug_kps is not None:
        aug_kps = aug_kps.clip_out_of_image()
    if aug_polys is not None:
        aug_polys = aug_polys.clip_out_of_image()

    stem = Path(img_path).stem
    aug_name = f"{stem}_aug_{idx}"
    aug_img_path = Path(out_dir) / f"{aug_name}.jpg"
    aug_json_path = Path(out_dir) / f"{aug_name}.json"

    cv2.imwrite(str(aug_img_path), aug_img)
    new_json = rebuild_labelme(aug_img, aug_bbs, aug_kps, aug_polys, shape_meta, f"{aug_name}.jpg")
    with open(aug_json_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)



def apply_aug(img_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for img_file in glob.glob(os.path.join(img_dir, "*.jpg")):
        json_file = Path(img_file).with_suffix(".json")
        if not json_file.exists():
            continue
        # 每张图做 2 次增强（可自行修改）
        for i in range(2):
            augment_data(img_file, json_file, dst_dir, i)


# ---------- 入口 ----------
if __name__ == "__main__":
    apply_aug(r"E:\demo\py\test01\1", r"E:\demo\py\test01\1\aug")
