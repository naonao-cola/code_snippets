from ultralytics import YOLO
from pathlib import Path
# https://blog.csdn.net/qq_37706472/article/details/128714604
weights = "/home/greatek/huangsc/test001/run/konghu2/weights/best.pt"
img_path = "/home/greatek/wangww/demo/py/test003/66e05ccc3f89422c9a0540097f4462ca.png"
out_dir = Path("/home/greatek/wangww/demo/py/test003/output/my_feats")          # 你想放的目录

model = YOLO(weights)

# 只改 project/name，让 YOLO 自动创建 output/my_feats/feature_maps/
model.predict(
    img_path,
    visualize=True,        # 打开特征图可视化
    project=out_dir.parent,  # 对应 output
    name=out_dir.name        # 对应 my_feats
)

print("特征图已保存到:", out_dir / "feature_maps")
