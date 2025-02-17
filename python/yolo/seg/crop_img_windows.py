import os
import cv2
import numpy as np
from pathlib import Path

def sliding_window(image, window_size, overlap_ratio):
    """
    生成滑动窗口。

    参数:
        image (numpy.ndarray): 输入图片。
        window_size (tuple): 滑动窗口的大小 (width, height)。
        overlap_ratio (float): 窗口之间的重叠比例（0 到 1）。

    返回:
        list: 滑动窗口的左上角坐标和裁剪图片。
    """
    windows = []
    h, w = image.shape[:2]
    window_w, window_h = window_size
    step_x = int(window_w * (1 - overlap_ratio))  # 计算步长
    step_y = int(window_h * (1 - overlap_ratio))

    x = 0
    while x < w:
        y = 0
        while y < h:
            # 计算窗口的右下角坐标
            x2 = x + window_w
            y2 = y + window_h

            # 超出图片边界时进行调整
            if x2 > w:
                x2 = w
                x = x2 - window_w
            if y2 > h:
                y2 = h
                y = y2 - window_h

            # 裁剪窗口
            window = image[y:y2, x:x2]
            windows.append(((x, y, x2, y2), window))

            y += step_y
            # 边界调整
            if y + window_h > h:
                break
        x += step_x
        # 边界调整
        if x + window_w > w:
            break

    return windows

def crop_images_and_polygons(image_folder, annotation_folder, output_image_folder, output_annotation_folder, window_size, overlap_ratio):
    """
    使用滑动窗口裁剪图片，并同步裁剪 YOLO 分割模型的多边形标注。

    参数:
        image_folder (Path): 包含图片的文件夹路径。
        annotation_folder (Path): 包含多边形标注的文件夹路径。
        output_image_folder (Path): 保存裁剪后的图片的文件夹路径。
        output_annotation_folder (Path): 保存裁剪后的多边形标注的文件夹路径。
        window_size (tuple): 滑动窗口的大小 (width, height)。
        overlap_ratio (float): 窗口之间的重叠比例（0 到 1）。
    """
    # 创建输出文件夹
    output_image_folder.mkdir(parents=True, exist_ok=True)
    output_annotation_folder.mkdir(parents=True, exist_ok=True)

    # 遍历图片文件夹中的图片
    for image_file in image_folder.glob('*.*'):
        # 获取图片文件名和扩展名
        image_name = image_file.stem
        image_ext = image_file.suffix

        # 读取图片
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Error: Unable to read {image_file}")
            continue

        # 获取对应的标注文件
        annotation_file = annotation_folder / f"{image_name}.txt"
        if not annotation_file.exists():
            print(f"Warning: No annotation found for {image_file}")
            continue

        # 读取标注文件
        with open(annotation_file, 'r') as f:
            annotations = [line.strip().split() for line in f.readlines()]

        # 滑动窗口裁剪
        windows = sliding_window(img, window_size, overlap_ratio)

        # 裁剪图片和标注
        for i, (window_coord, window_image) in enumerate(windows):
            # 生成唯一的输出文件名
            output_image_name = f"{image_name}_window{i}{image_ext}"
            output_annotation_name = f"{image_name}_window{i}.txt"

            # 保存裁剪后的图片
            output_image_path = output_image_folder / output_image_name
            cv2.imwrite(str(output_image_path), window_image)

            # 裁剪多边形标注
            new_annotations = []
            window_x1, window_y1, window_x2, window_y2 = window_coord
            window_w = window_x2 - window_x1
            window_h = window_y2 - window_y1

            for ann in annotations:
                # 假设标注格式为 <class_id> x1 y1 x2 y2 ... xn yn
                class_id = ann[0]
                polygon = list(map(float, ann[1:]))

                # 转换为绝对坐标
                abs_polygon = []
                for j in range(0, len(polygon), 2):
                    x = polygon[j] * img.shape[1]
                    y = polygon[j+1] * img.shape[0]
                    abs_polygon.extend([x, y])

                old_max_x = max(abs_polygon[j] for j in range(0, len(abs_polygon), 2))
                old_max_y = max(abs_polygon[j+1] for j in range(0, len(abs_polygon), 2))
                old_min_x = min(abs_polygon[j] for j in range(0, len(abs_polygon), 2))
                old_min_y = min(abs_polygon[j+1] for j in range(0, len(abs_polygon), 2))

                # 判断多边形是否在当前窗口内
                # 这里简单地检查至少有一个点在窗口内
                in_window = False
                in_side = False
                for j in range(0, len(abs_polygon), 2):
                    x = abs_polygon[j]
                    y = abs_polygon[j+1]
                    # 在图像边边的不要
                    if old_max_x >= window_x2  or  old_min_x <= window_x1  or old_min_y <= window_y1 or old_max_y >= window_y2:
                        in_window = False
                        in_side = True
                        break
                    if window_x1 <= x <= window_x2 and window_y1 <= y <= window_y2  and not in_side:
                        in_window = True
                        break
                if not in_window:
                    continue  # 多边形完全在窗口外，跳过

                # 调整多边形到窗口内的坐标
                cropped_polygon = []
                for j in range(0, len(abs_polygon), 2):
                    x = abs_polygon[j]
                    y = abs_polygon[j+1]
                    new_x = max(x - window_x1, 0)
                    new_x = min(new_x, window_w)
                    new_y = max(y - window_y1, 0)
                    new_y = min(new_y, window_h)
                    cropped_polygon.extend([new_x, new_y])

                # 将多边形转换为 YOLO 格式（归一化坐标）
                new_polygon = []
                for j in range(0, len(cropped_polygon), 2):
                    new_x = cropped_polygon[j] / window_w
                    new_y = cropped_polygon[j+1] / window_h
                    new_polygon.extend([new_x, new_y])

                # 保存新的标注信息
                new_ann = [class_id] + new_polygon
                new_annotations.append(new_ann)

            # 保存裁剪后的标注文件
            output_annotation_path = output_annotation_folder / output_annotation_name
            with open(output_annotation_path, 'w') as f:
                for ann in new_annotations:
                    f.write(' '.join(map(str, ann)) + '\n')

            print(f"Processed {output_image_name} and saved annotation {output_annotation_name}")

# 示例用法
if __name__ == "__main__":
    image_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/images/")  # 替换为您的图片文件夹路径
    annotation_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/labels/")  # 替换为您的标注文件夹路径
    output_image_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216_1/images/")  # 替换为图片输出文件夹路径
    output_annotation_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216_1/labels/")  # 替换为标注输出文件夹路径
    window_size = (400, 400)  # 滑动窗口的大小
    overlap_ratio = 0.5  # 窗口之间的重叠比例

    crop_images_and_polygons(image_folder, annotation_folder, output_image_folder, output_annotation_folder, window_size, overlap_ratio)