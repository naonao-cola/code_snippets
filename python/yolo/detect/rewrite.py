import glob
import cv2
# 消除数据警告
# img_file = glob.glob(
#     "/data/proj/www/repo/yolo8_test/dataset/lego/images/*.jpg")

# print("图片数量： ", len(img_file))
# for item in img_file:
#     cur_img = cv2.imread(item)
#     cv2.imwrite(item, cur_img)


from pathlib2 import Path


def fix_jpg():
    img_file = glob.glob(
        "/data/proj/www/repo/yolo8_test/dataset/digita/*.JPG")
    for item in img_file:
        before_path = Path(item)
        after_path = before_path.with_suffix('.jpg')
        before_path.rename(after_path)


fix_jpg()
