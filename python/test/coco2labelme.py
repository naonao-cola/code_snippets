import argparse
import json
import os
from pycocotools.coco import COCO


def segmentation_to_shapes(segmentation, label, bbox=None):
    """将COCO分割格式转换为LabelMe的shape格式"""
    shapes = []
    # 处理多边形分割
    if isinstance(segmentation, list) and len(segmentation) > 0 and len(segmentation[0]) >= 6:
        for seg in segmentation:
            points = [[float(seg[i]), float(seg[i + 1])] for i in range(0, len(seg), 2)]
            shapes.append(
                {"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}
            )
    # 处理RLE分割或仅有bbox的情况
    elif bbox is not None:
        x, y, w, h = map(float, bbox)
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        shapes.append(
            {"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}
        )
    return shapes


def coco_to_labelme(coco_path, ann_file=None, output_dir=None):
    """核心转换函数"""
    # 自动查找标注文件（精简查找逻辑）
    if not ann_file:
        ann_candidates = [
            os.path.join(coco_path, "annotations.json"),
            os.path.join(coco_path, "annotations", "instances_train.json"),
            os.path.join(coco_path, "annotations", "instances_val.json"),
        ]
        ann_file = next((p for p in ann_candidates if os.path.exists(p)), None)
        if not ann_file:
            raise FileNotFoundError("未找到COCO标注文件，请指定--ann-file")

    # 默认输出目录
    output_dir = output_dir or os.path.join(coco_path, "labelme_jsons")
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO数据并转换
    coco = COCO(ann_file)
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        shapes = []
        for ann in anns:
            label = coco.loadCats(ann["category_id"])[0]["name"]
            shapes.extend(segmentation_to_shapes(ann.get("segmentation"), label, ann.get("bbox")))

        # 构建LabelMe JSON
        labelme_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_info["file_name"],
            "imageData": None,
            "imageHeight": img_info["height"],
            "imageWidth": img_info["width"],
        }

        # 保存文件
        json_name = os.path.splitext(img_info["file_name"])[0] + ".json"
        with open(os.path.join(output_dir, json_name), "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        print(f"转换完成: {img_info['file_name']} -> {json_name} (共{len(shapes)}个标注)")

    print(f"\n所有文件转换完成！输出目录: {output_dir}")


if __name__ == "__main__":
    # 配置默认参数，简化命令行解析
    parser = argparse.ArgumentParser(description="COCO标注转LabelMe JSON (精简版)")
    parser.add_argument(
        "coco_path",
        nargs="?",
        default=r"F:\data\---Uploaded on 04-08-26 at 3-15 pm.coco\train",
        help="COCO数据集路径（默认使用预设路径）",
    )
    parser.add_argument("--ann-file", default=None, help="标注文件路径（默认自动查找）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认: coco_path/labelme_jsons）")
    args = parser.parse_args()

    # 执行转换
    coco_to_labelme(args.coco_path, args.ann_file, args.output_dir)
