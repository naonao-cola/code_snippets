import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int


def _try_get_image_size_with_pillow(image_path: Path) -> ImageInfo | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        with Image.open(image_path) as im:
            width, height = im.size
        return ImageInfo(width=int(width), height=int(height))
    except Exception:
        return None


def _get_png_size(fp) -> ImageInfo | None:
    header = fp.read(24)
    if len(header) < 24:
        return None
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return ImageInfo(width=int(width), height=int(height))


def _get_bmp_size(fp) -> ImageInfo | None:
    header = fp.read(26)
    if len(header) < 26:
        return None
    if header[:2] != b"BM":
        return None
    width, height = struct.unpack("<ii", header[18:26])
    return ImageInfo(width=int(width), height=abs(int(height)))


def _get_jpeg_size(fp) -> ImageInfo | None:
    data = fp.read(2)
    if data != b"\xff\xd8":
        return None

    while True:
        marker_start = fp.read(1)
        if not marker_start:
            return None
        while marker_start != b"\xff":
            marker_start = fp.read(1)
            if not marker_start:
                return None
        while True:
            marker_type = fp.read(1)
            if not marker_type:
                return None
            if marker_type != b"\xff":
                break

        if marker_type in {b"\xd8", b"\xd9"}:
            continue

        length_bytes = fp.read(2)
        if len(length_bytes) != 2:
            return None
        (segment_length,) = struct.unpack(">H", length_bytes)
        if segment_length < 2:
            return None

        if marker_type in {
            b"\xc0",
            b"\xc1",
            b"\xc2",
            b"\xc3",
            b"\xc5",
            b"\xc6",
            b"\xc7",
            b"\xc9",
            b"\xca",
            b"\xcb",
            b"\xcd",
            b"\xce",
            b"\xcf",
        }:
            seg = fp.read(5)
            if len(seg) != 5:
                return None
            precision = seg[0]
            if precision == 0:
                return None
            height, width = struct.unpack(">HH", seg[1:5])
            return ImageInfo(width=int(width), height=int(height))

        fp.seek(segment_length - 2, os.SEEK_CUR)


def get_image_size(image_path: Path) -> ImageInfo:
    pillow_info = _try_get_image_size_with_pillow(image_path)
    if pillow_info is not None:
        return pillow_info

    with open(image_path, "rb") as fp:
        fp.seek(0)
        for reader in (_get_png_size, _get_jpeg_size, _get_bmp_size):
            fp.seek(0)
            info = reader(fp)
            if info is not None:
                return info

    raise ValueError(f"无法解析图片尺寸: {image_path}")


def load_class_names(classes_file: Path | None) -> list[str] | None:
    if classes_file is None:
        return None
    if not classes_file.exists():
        raise FileNotFoundError(f"classes 文件不存在: {classes_file}")
    names: list[str] = []
    with open(classes_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip().lstrip("\ufeff")
            if name:
                names.append(name)
    return names if names else None


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def parse_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    items: list[tuple[int, float, float, float, float]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            items.append((class_id, x, y, w, h))
    return items


def yolo_xywh_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if max(x, y, w, h) <= 1.0:
        x *= image_width
        w *= image_width
        y *= image_height
        h *= image_height

    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    x1 = clamp(x1, 0.0, float(image_width))
    y1 = clamp(y1, 0.0, float(image_height))
    x2 = clamp(x2, 0.0, float(image_width))
    y2 = clamp(y2, 0.0, float(image_height))
    return x1, y1, x2, y2


def find_image_for_label(root: Path, label_path: Path) -> Path | None:
    try:
        rel = label_path.relative_to(root)
    except Exception:
        rel = None

    candidates: list[Path] = []
    if rel is not None and len(rel.parts) >= 2 and rel.parts[0].lower() == "labels":
        rel_no_ext = rel.with_suffix("")
        image_rel_no_ext = Path("images", *rel_no_ext.parts[1:])
        for ext in IMAGE_EXTS:
            candidates.append(root / (str(image_rel_no_ext) + ext))
    else:
        stem = label_path.with_suffix("").name
        for ext in IMAGE_EXTS:
            candidates.append(label_path.with_name(stem + ext))
            candidates.append(root / (stem + ext))

    for p in candidates:
        if p.exists():
            return p
    return None


def build_labelme_json(
    image_path: Path,
    image_info: ImageInfo,
    yolo_items: list[tuple[int, float, float, float, float]],
    class_names: list[str] | None,
    image_path_in_json: str,
) -> dict:
    shapes: list[dict] = []
    for class_id, x, y, w, h in yolo_items:
        x1, y1, x2, y2 = yolo_xywh_to_xyxy(x, y, w, h, image_info.width, image_info.height)
        label = str(class_id)
        if class_names is not None and 0 <= class_id < len(class_names):
            label = class_names[class_id]
        shapes.append(
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
        )

    return {
        "version": "5.5.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path_in_json).name,
        "imageData": None,
        "imageHeight": image_info.height,
        "imageWidth": image_info.width,
    }


def write_json(json_path: Path, data: dict, overwrite: bool) -> bool:
    if json_path.exists() and not overwrite:
        return False
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def collect_label_files(root: Path, subset: str) -> list[Path]:
    label_root = root / "labels"
    if label_root.exists():
        base = label_root
    else:
        base = root

    if subset != "all":
        base = base / subset
        if not base.exists():
            return []

    label_files: list[Path] = []
    for dirpath, _, filenames in os.walk(base):
        for name in filenames:
            if not name.lower().endswith(".txt"):
                continue
            if name.lower().endswith(".cache"):
                continue
            label_files.append(Path(dirpath) / name)
    label_files.sort()
    return label_files


def main() -> int:
    default_root = r"F:\data\指针仪表数据集\指针仪表数据集\指针仪表检测\coco\origin"

    parser = argparse.ArgumentParser(description="YOLOv8(txt) 转 LabelMe(json)")
    parser.add_argument("--root", default=default_root, help="YOLO 数据集根目录（包含 images/ labels/）")
    parser.add_argument("--subset", default="val", choices=["train", "val", "all"], help="只转换哪个子集")
    parser.add_argument("--limit", type=int, default=0, help="只转换前 N 张；0 表示不限制")
    parser.add_argument("--out-dir", default=r"e:\demo\py\txt2json\labelme_out2", help="输出目录；留空表示 json 写在图片旁边")
    parser.add_argument("--classes", default=r"E:\demo\py\txt2json\classes.txt", help="类别名文件，每行一个类别；留空则用数字类别")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 json")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    classes_file = Path(args.classes).expanduser().resolve() if args.classes else None
    class_names = load_class_names(classes_file)

    label_files = collect_label_files(root, args.subset)
    if not label_files:
        print(f"未找到标签文件: root={root} subset={args.subset}")
        return 2

    converted = 0
    skipped_no_image = 0
    skipped_write = 0
    failed = 0

    for label_path in label_files:
        if args.limit > 0 and converted >= args.limit:
            break

        image_path = find_image_for_label(root, label_path)
        if image_path is None:
            skipped_no_image += 1
            continue

        try:
            image_info = get_image_size(image_path)
        except Exception as e:
            print(f"[失败] 读图片尺寸失败: {image_path} ({e})")
            failed += 1
            continue

        yolo_items = parse_yolo_label_file(label_path)

        if out_dir is None:
            json_path = image_path.with_suffix(".json")
        else:
            rel_from_images = None
            try:
                rel_from_images = image_path.relative_to(root / "images")
            except Exception:
                rel_from_images = Path(image_path.name)
            json_path = (out_dir / rel_from_images).with_suffix(".json")
        image_path_in_json = image_path.name

        data = build_labelme_json(
            image_path=image_path,
            image_info=image_info,
            yolo_items=yolo_items,
            class_names=class_names,
            image_path_in_json=image_path_in_json,
        )

        if args.dry_run:
            print(f"[预览] {label_path} -> {json_path}")
            converted += 1
            continue

        try:
            wrote = write_json(json_path, data, overwrite=args.overwrite)
            if wrote:
                print(f"[OK] {image_path} -> {json_path}")
                converted += 1
            else:
                skipped_write += 1
        except Exception as e:
            print(f"[失败] 写 json 失败: {json_path} ({e})")
            failed += 1

    print(
        f"完成: converted={converted}, skipped_no_image={skipped_no_image}, "
        f"skipped_exists={skipped_write}, failed={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
