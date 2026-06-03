import argparse
import json
import os
from pathlib import Path


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


def read_labelme_json(json_path: Path) -> dict:
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(json_path.read_text(encoding="utf-8-sig"))


def shape_to_xyxy(shape: dict) -> tuple[float, float, float, float] | None:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for p in points:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
        except Exception:
            continue
    if not xs or not ys:
        return None
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    return x1, y1, x2, y2


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    x1 = clamp(x1, 0.0, float(image_width))
    y1 = clamp(y1, 0.0, float(image_height))
    x2 = clamp(x2, 0.0, float(image_width))
    y2 = clamp(y2, 0.0, float(image_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0

    if image_width <= 0 or image_height <= 0:
        raise ValueError("imageWidth/imageHeight 非法")

    return (
        xc / float(image_width),
        yc / float(image_height),
        w / float(image_width),
        h / float(image_height),
    )


def label_to_class_id(label: str, name_to_id: dict[str, int] | None) -> int | None:
    lab = label.strip().lstrip("\ufeff")
    if not lab:
        return None
    if name_to_id is not None and lab in name_to_id:
        return name_to_id[lab]
    try:
        return int(float(lab))
    except Exception:
        return None


def convert_one(json_path: Path, out_txt_path: Path, name_to_id: dict[str, int] | None) -> tuple[int, int]:
    data = read_labelme_json(json_path)
    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        shapes = []

    lines: list[str] = []
    kept = 0
    skipped = 0

    for shape in shapes:
        if not isinstance(shape, dict):
            skipped += 1
            continue
        label = shape.get("label")
        if not isinstance(label, str):
            skipped += 1
            continue

        class_id = label_to_class_id(label, name_to_id)
        if class_id is None:
            skipped += 1
            continue

        xyxy = shape_to_xyxy(shape)
        if xyxy is None:
            skipped += 1
            continue

        try:
            xc, yc, w, h = xyxy_to_yolo(*xyxy, image_width=image_width, image_height=image_height)
        except Exception:
            skipped += 1
            continue

        xc = clamp(xc, 0.0, 1.0)
        yc = clamp(yc, 0.0, 1.0)
        w = clamp(w, 0.0, 1.0)
        h = clamp(h, 0.0, 1.0)
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        kept += 1

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return kept, skipped


def collect_json_files(in_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(in_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe(json) 转 YOLOv8(txt)")
    parser.add_argument("--in-dir", default=r"F:\data\指针仪表数据集\指针仪表数据集\指针仪表检测\coco\train2017", help="输入 json 目录（递归搜索）")
    parser.add_argument("--out-dir", default=r"e:\demo\py\txt2json\yolo_labels_txt_out", help="输出 txt 目录")
    parser.add_argument("--classes", default=r"e:\demo\py\txt2json\classes.txt", help="类别名文件（每行一个类别）")
    parser.add_argument("--limit", type=int, default=0, help="只转换前 N 个文件；0 表示全部")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 txt")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    classes_file = Path(args.classes).expanduser().resolve() if args.classes else None
    class_names = load_class_names(classes_file)
    name_to_id = {name: i for i, name in enumerate(class_names)} if class_names else None

    if not in_dir.exists():
        print(f"输入目录不存在: {in_dir}")
        return 2

    json_files = collect_json_files(in_dir)
    if not json_files:
        print(f"未找到 json: {in_dir}")
        return 2

    converted = 0
    failed = 0
    total_kept = 0
    total_skipped = 0

    for json_path in json_files:
        if args.limit and converted >= args.limit:
            break

        rel = json_path.relative_to(in_dir)
        out_txt_path = (out_dir / rel).with_suffix(".txt")

        if out_txt_path.exists() and not args.overwrite:
            continue

        if args.dry_run:
            print(f"[预览] {json_path} -> {out_txt_path}")
            converted += 1
            continue

        try:
            kept, skipped = convert_one(json_path, out_txt_path, name_to_id)
            total_kept += kept
            total_skipped += skipped
            converted += 1
            print(f"[OK] {json_path} -> {out_txt_path} (kept={kept}, skipped={skipped})")
        except Exception as e:
            failed += 1
            print(f"[失败] {json_path} ({e})")

    print(f"完成: converted={converted}, failed={failed}, kept={total_kept}, skipped={total_skipped}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
