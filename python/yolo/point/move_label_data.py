from pathlib import Path
import shutil

src_label_dir = Path(r'/home/greatek/wangww/datasets/point/labels/train/')   # 原标注目录
dst_label_dir = Path(r'/home/greatek/wangww/datasets/point/labels/val/')     # 目标标注目录
val_img_dir   = Path(r'/home/greatek/wangww/datasets/point/images/val/')     # 验证集图片目录

dst_label_dir.mkdir(parents=True, exist_ok=True)

for img_path in val_img_dir.glob('*.jpg'):  # 如有其它后缀再改
    txt_name = img_path.with_suffix('.txt').name
    txt_src  = src_label_dir / txt_name
    txt_dst  = dst_label_dir / txt_name
    if txt_src.exists():
        shutil.move(str(txt_src), str(txt_dst))

print('验证集标注移动完成！')