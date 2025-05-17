import os
import cv2
from tqdm import tqdm
# yolo det采用保持纵横比的方式resize图像后,进行pad后得道输入图像,以下逻辑与yolo预处理一致
target_size = (1920, 1920)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def resize_img_for_quantization(img_dir, save_dir):
    for filename in tqdm(os.listdir(img_dir)) :
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        img, ratio, _ = letterbox(img, target_size)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)



if __name__ == '__main__':
    img_dir = r"/home/proj/www/py/data/pic_org"  # 原始图像目录
    save_dir = r"/home/proj/www/py/data/after"  # 处理后图像目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    resize_img_for_quantization(img_dir, save_dir)