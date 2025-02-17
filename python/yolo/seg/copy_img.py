import os
import shutil
from pathlib import Path

def duplicate_images_and_annotations(image_folder, annotation_folder, output_folder, num_copies):
    """
    将图片和标注文件分别从两个文件夹中读取，并复制多份到指定的输出文件夹。

    参数:
        image_folder (Path): 包含图片文件的文件夹路径。
        annotation_folder (Path): 包含标注文件的文件夹路径。
        output_folder (Path): 保存复制后文件的输出文件夹路径。
        num_copies (int): 文件需要复制的份数。
    """
    # 确保输出文件夹存在
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取图片文件列表
    image_files = list(image_folder.glob('*'))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    # 获取标注文件列表
    annotation_files = list(annotation_folder.glob('*'))
    annotation_files = [f for f in annotation_files if f.is_file() and f.suffix.lower() in {'.json', '.txt'}]

    # 检查图片文件和标注文件数量是否一致
    if len(image_files) != len(annotation_files):
        raise ValueError("图片文件和标注文件数量不一致")

    # 复制文件
    for img_file in image_files:
        for ann_file in annotation_files:
            # 确保图片文件和标注文件的文件名一致
            if img_file.stem != ann_file.stem:
                print(f"Skipped due to inconsistent names: {img_file.name} and {ann_file.name}")
                continue

            for i in range(num_copies):
                # 生成新的文件名
                new_img_name = f"{img_file.stem}_copy{i+1}{img_file.suffix}"
                new_ann_name = f"{ann_file.stem}_copy{i+1}{ann_file.suffix}"

                # 构建输出文件路径
                output_img = output_folder / new_img_name
                output_ann = output_folder / new_ann_name

                # 复制图片文件
                shutil.copy2(img_file, output_img)
                print(f"Copied {img_file.name} to {output_img}")

                # 复制标注文件
                shutil.copy2(ann_file, output_ann)
                print(f"Copied {ann_file.name} to {output_ann}")

# 示例用法
if __name__ == "__main__":
    # 使用 Path 对象指定路径
    image_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/images/")  # 替换为你的图片文件夹路径
    annotation_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/labels/")  # 替换为你的标注文件夹路径
    output_folder = Path("/home/tvt/luobing/hf_dataset/jinzhen_seg/0216/ooo/")  # 替换为你的输出文件夹路径
    num_copies = 8  # 替换为你需要的复制份数

    duplicate_images_and_annotations(image_folder, annotation_folder, output_folder, num_copies)