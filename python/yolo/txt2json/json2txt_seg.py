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


def label_to_class_id(label: str, name_to_id: dict[str, int] | None, fallback_zero: bool) -> int | None:
    lab = label.strip().lstrip("\ufeff")
    if not lab:
        return None
    if name_to_id is not None and lab in name_to_id:
        return name_to_id[lab]
    try:
        return int(float(lab))
    except Exception:
        return 0 if fallback_zero else None


def normalize_points(points: list[list[float]], image_width: int, image_height: int) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        x = clamp(x, 0.0, float(image_width)) / float(image_width)
        y = clamp(y, 0.0, float(image_height)) / float(image_height)
        out.append((x, y))
    return out


def shape_to_polygon_points(shape: dict) -> list[list[float]] | None:
    points = shape.get("points")
    if not isinstance(points, list) or not points:
        return None

    try:
        shape_type = str(shape.get("shape_type") or "")
    except Exception:
        shape_type = ""

    if shape_type == "rectangle" and len(points) >= 2:
        p1 = points[0]
        p2 = points[1]
        if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) >= 2 and len(p2) >= 2):
            return None
        try:
            x1 = float(p1[0])
            y1 = float(p1[1])
            x2 = float(p2[0])
            y2 = float(p2[1])
        except Exception:
            return None
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return points if len(points) >= 3 else None


def convert_one(
    json_path: Path,
    out_txt_path: Path,
    name_to_id: dict[str, int] | None,
    fallback_zero: bool,
) -> tuple[int, int, int]:
    data = read_labelme_json(json_path)
    image_width = int(data.get("imageWidth") or 0)
    image_height = int(data.get("imageHeight") or 0)
    if image_width <= 0 or image_height <= 0:
        raise ValueError("imageWidth/imageHeight 非法")

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        shapes = []

    lines: list[str] = []
    kept = 0
    skipped = 0
    unknown_label = 0

    for shape in shapes:
        if not isinstance(shape, dict):
            skipped += 1
            continue

        label = shape.get("label")
        if not isinstance(label, str):
            skipped += 1
            continue

        class_id = label_to_class_id(label, name_to_id, fallback_zero=fallback_zero)
        if class_id is None:
            skipped += 1
            unknown_label += 1
            continue

        polygon = shape_to_polygon_points(shape)
        if polygon is None:
            skipped += 1
            continue

        norm_pts = normalize_points(polygon, image_width=image_width, image_height=image_height)
        if len(norm_pts) < 3:
            skipped += 1
            continue

        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_pts)
        lines.append(f"{class_id} {flat}")
        kept += 1

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return kept, skipped, unknown_label


def collect_json_files(in_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(in_dir):
        for name in filenames:
            if name.lower().endswith(".json"):
                files.append(Path(dirpath) / name)
    files.sort()
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="LabelMe(json) 转 YOLOv8-SEG(txt)")
    parser.add_argument("--in-dir", default=r"F:\wangguanglei\训练数据", help="输入目录（包含 json；递归搜索）")
    parser.add_argument("--out-dir", default=r"F:\wangguanglei\训练数据\labels", help="输出 labels 目录（生成 .txt）")  
    parser.add_argument("--classes", default=r"e:\demo\py\txt2json\classes.txt", help="类别名文件（每行一个类别）")
    parser.add_argument("--limit", type=int, default=0, help="只转换前 N 个文件；0 表示全部")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 txt")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    parser.add_argument("--strict", action="store_true", help="严格模式：classes 里找不到标签名就跳过")
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    classes_file = Path(args.classes).expanduser().resolve() if args.classes else None
    class_names = load_class_names(classes_file) if args.classes else None
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
    total_unknown = 0

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
            kept, skipped, unknown = convert_one(
                json_path,
                out_txt_path,
                name_to_id=name_to_id,
                fallback_zero=not args.strict,
            )
            total_kept += kept
            total_skipped += skipped
            total_unknown += unknown
            converted += 1
            print(f"[OK] {json_path} -> {out_txt_path} (kept={kept}, skipped={skipped})")
        except Exception as e:
            failed += 1
            print(f"[失败] {json_path} ({e})")

    print(
        f"完成: converted={converted}, failed={failed}, kept={total_kept}, "
        f"skipped={total_skipped}, unknown_label={total_unknown}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
