import cv2, json, os, random, glob
from pathlib import Path
import albumentations as A


# ---------- 1. 把 labelme 解析成 albumentations 需要的格式 ----------
def parse_labelme(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bboxes, keypoints, polygons, shape_meta = [], [], [], []

    for shape in data["shapes"]:
        label = shape["label"]
        pts = shape["points"]

        if shape["shape_type"] == "rectangle":
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x1, x2 = sorted([x1, x2])   # 谁小谁当左边
            y1, y2 = sorted([y1, y2])   # 谁小谁当顶边
            bboxes.append([x1, y1, x2, y2])  # [x1,y1,x2,y2]
            shape_meta.append(shape)

        elif shape["shape_type"] == "point":
            x, y = pts[0]
            keypoints.append([x, y])  # [x,y]
            shape_meta.append(shape)

        elif shape["shape_type"] == "polygon":
            polygons.append(pts)  # [[x,y], ...]
            shape_meta.append(shape)

    return bboxes, keypoints, polygons, shape_meta, data


# ---------- 2. 把增强后的结果写回 labelme ----------
def rebuild_labelme(aug, bboxes_aug, keypoints_aug, polygons_aug, shape_meta, imagePath):
    new_shapes, bb_idx, kp_idx, poly_idx = [], 0, 0, 0

    for shape in shape_meta:
        new_shape = shape.copy()
        if shape["shape_type"] == "rectangle":
            x1, y1, x2, y2 = bboxes_aug[bb_idx]
            bb_idx += 1
            new_shape["points"] = [[float(x1), float(y1)], [float(x2), float(y2)]]

        elif shape["shape_type"] == "point":
            x, y = keypoints_aug[kp_idx]
            kp_idx += 1
            new_shape["points"] = [[float(x), float(y)]]

        elif shape["shape_type"] == "polygon":
            pts = polygons_aug[poly_idx]
            poly_idx += 1
            new_shape["points"] = [[float(x), float(y)] for x, y in pts]

        new_shapes.append(new_shape)

    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": new_shapes,
        "imagePath": imagePath,
        "imageData": None,
        "imageHeight": aug["image"].shape[0],
        "imageWidth": aug["image"].shape[1],
    }


# ---------- 3. 构建增强流水线 ----------
def get_augmentor(p=0.5):
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=p * 0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=p * 0.5),
            A.GaussNoise(var_limit=(0, 0.05 * 255), p=p * 0.5),
            A.HorizontalFlip(p=0.1),
            A.RandomRotate90(p=0.1),
            A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), p=p),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(format="xy", label_fields=["kp_labels"]),
        additional_targets={"polygon": "polygon"},
    )


# ---------- 4. 单张增强 ----------
def augment_data(img_path, json_path, out_dir, idx):
    img = cv2.imread(img_path)
    bboxes, keypoints, polygons, shape_meta, _ = parse_labelme(json_path)

    # 给 albumentations 准备标签（占位即可）
    bbox_labels = ["obj"] * len(bboxes)
    kp_labels = ["pt"] * len(keypoints)

    aug = get_augmentor()(
        image=img,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        kp_labels=kp_labels,
        polygon=polygons,
    )

    stem = Path(img_path).stem
    aug_name = f"{stem}_aug_{idx}"
    aug_img_path = Path(out_dir) / f"{aug_name}.jpg"
    aug_json_path = Path(out_dir) / f"{aug_name}.json"

    cv2.imwrite(str(aug_img_path), aug["image"])
    new_json = rebuild_labelme(
        aug,
        bboxes_aug=aug["bboxes"],
        keypoints_aug=aug["keypoints"],
        polygons_aug=aug["polygon"],
        shape_meta=shape_meta,
        imagePath=f"{aug_name}.jpg",
    )
    with open(aug_json_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=2)


# ---------- 5. 批处理 ----------
def apply_aug(img_dir, dst_dir, times=2):
    os.makedirs(dst_dir, exist_ok=True)
    for img_file in glob.glob(os.path.join(img_dir, "*.jpg")):
        json_file = Path(img_file).with_suffix(".json")
        if not json_file.exists():
            continue
        for i in range(times):
            augment_data(img_file, json_file, dst_dir, i)


# ---------- 6. 入口 ----------
if __name__ == "__main__":
    apply_aug(r"E:\demo\py\test01\1", r"E:\demo\py\test01\1\aug", times=2)
