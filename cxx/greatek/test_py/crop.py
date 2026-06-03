import json
import shutil
from pathlib import Path
from PIL import Image


def crop_by_label(json_path, src_image_dir, dst_crop_dir, target_label):
    json_path = Path(json_path)
    src_image_dir = Path(src_image_dir)
    dst_crop_dir = Path(dst_crop_dir)
    dst_crop_dir.mkdir(parents=True, exist_ok=True)

    # 读取 Labelme JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_name = data["imagePath"]
    image_path = src_image_dir / image_name

    if not image_path.exists():
        print(f"⚠️ 图片不存在：{image_path}")
        return

    img = Image.open(image_path).convert("RGB")

    crop_count = 0
    for shape in data["shapes"]:
        label = shape["label"]
        if label != target_label:
            continue

        points = shape["points"]
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        cropped = img.crop((x_min, y_min, x_max, y_max))

        # 保存裁剪图
        crop_name = f"{json_path.stem}_{crop_count}.jpg"
        crop_path = dst_crop_dir / crop_name
        cropped.save(crop_path)
        crop_count += 1
        print(f"✅ 保存裁剪图：{crop_path}")


# ✅ 使用示例
if __name__ == "__main__":
    json_dir = Path(r"F:\data\use\use")
    image_dir = Path(r"F:\data\use\use")
    output_dir = Path(r"F:\data\use\biao")
    label_to_crop = "clear"  # 替换为你想裁剪的标签

    for json_file in json_dir.glob("*.json"):
        crop_by_label(json_file, image_dir, output_dir, label_to_crop)
