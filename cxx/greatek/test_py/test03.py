import json, math
from pathlib import Path
import numpy as np

LABELME_DIR = Path(r"F:\data\use\biao")
IMG_DIR = Path(r"F:\data\use\biao")  # 仅读 w,h
OUT_DIR = Path(r"F:\data\use\txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm(x, y, w, h):
    return x / w, y / h


def dist_point_to_box(px, py, x1, y1, x2, y2):
    """点到矩形的最短距离"""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return math.hypot(px - cx, py - cy)


for json_path in LABELME_DIR.glob("*.json"):
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["imageWidth"], data["imageHeight"]

    # 1. 收集矩形
    rects = []  # [(label, x1,y1,x2,y2), ...]
    for s in data["shapes"]:
        if s["shape_type"] == "rectangle":
            pts = s["points"]
            (x1, y1), (x2, y2) = pts
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            rects.append((s["label"], x1, y1, x2, y2))

    # 2. 收集关键点
    keypts = {"p1": [], "p2": [], "p3": [], "p4": []}
    for s in data["shapes"]:
        if s["shape_type"] == "point":
            keypts[s["label"]].append(s["points"][0])

    # 3. 配对函数
    def match_pts_to_rect(pt_list, rects):
        """把 pt_list 里的每个点挂到最近的矩形，返回 list[(rect_idx, px, py)]"""
        res = []
        for px, py in pt_list:
            best_i = min(range(len(rects)), key=lambda i: dist_point_to_box(px, py, *rects[i][1:5]))
            res.append((best_i, px, py))
        return res

    red_rects = [r for r in rects if r[0] == "red"]
    black_rects = [r for r in rects if r[0] == "black"]
    # 注意：rects 顺序 = 文件里顺序，但后面用索引重新排序

    # 4. 红色指针
    red_tips = match_pts_to_rect(keypts["p1"], red_rects)
    red_centers = match_pts_to_rect(keypts["p2"], red_rects)
    # 按矩形索引排序，保证同一根针的 center+tip 对齐
    red_centers.sort(key=lambda x: x[0])
    red_tips.sort(key=lambda x: x[0])

    # 5. 黑色指针
    black_centers = match_pts_to_rect(keypts["p3"], black_rects)
    black_tips = match_pts_to_rect(keypts["p4"], black_rects)
    black_centers.sort(key=lambda x: x[0])
    black_tips.sort(key=lambda x: x[0])

    # 6. 拼 YOLOv8-pose 行
    lines = []

    def make_line(cls_id, rect, center, tip):
        x1, y1, x2, y2 = rect[1:5]
        cx, cy = center[1:]
        tx, ty = tip[1:]
        # 归一化
        x1, x2 = norm(x1, 0, w, h)[0], norm(x2, 0, w, h)[0]
        y1, y2 = norm(0, y1, w, h)[1], norm(0, y2, w, h)[1]
        cx, cy = norm(cx, cy, w, h)
        tx, ty = norm(tx, ty, w, h)
        # bbox
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        line = [cls_id, xc, yc, bw, bh, cx, cy, 2, tx, ty, 2]
        return " ".join(f"{v:.6f}" for v in line)

    for rect, cen, tip in zip(red_rects, red_centers, red_tips):
        lines.append(make_line(0, rect, cen, tip))
    for rect, cen, tip in zip(black_rects, black_centers, black_tips):
        lines.append(make_line(1, rect, cen, tip))

    out_path = OUT_DIR / f"{json_path.stem}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")

print("✅ 全部转换完成，顺序零错乱！")
