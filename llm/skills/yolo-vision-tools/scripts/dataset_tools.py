#!/usr/bin/env python3
"""
YOLO Dataset Tools

This script provides tools for dataset preparation, format conversion, and analysis.
Extracted from dataset_preparation.md to save token usage.
"""

import json
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
# from sklearn.model_selection import train_test_split  # Moved inside function to handle optional dependency
# import yaml  # Moved inside function to handle optional dependency

# ============================================================================
# ADVANCED DATA PROCESSING (实战增强)
# ============================================================================

def sliding_window_crop(image_path, annotation_path, output_img_dir, output_ann_dir, window_size=(400, 400), overlap_ratio=0.5):
    """
    [实战经验] 大图滑动窗口裁剪 (支持多边形/分割标注)
    用于处理高分辨率图像（如遥感、工业缺陷），避免直接 resize 导致小目标丢失。
    
    Args:
        image_path: 原始图像路径
        annotation_path: 原始标注文件路径 (YOLO 格式)
        output_img_dir: 裁剪图像输出目录
        output_ann_dir: 裁剪标注输出目录
        window_size: 窗口大小 (width, height)
        overlap_ratio: 窗口重叠比例 (0-1)
    """
    import os
    import cv2
    from pathlib import Path
    
    image_file = Path(image_path)
    annotation_file = Path(annotation_path)
    output_img_dir = Path(output_img_dir)
    output_ann_dir = Path(output_ann_dir)
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = image_file.stem
    image_ext = image_file.suffix

    img = cv2.imread(str(image_file))
    if img is None:
        print(f"Error: Unable to read {image_file}")
        return

    if not annotation_file.exists():
        print(f"Warning: No annotation found for {image_file}")
        return

    with open(annotation_file, 'r') as f:
        annotations = [line.strip().split() for line in f.readlines()]

    h, w = img.shape[:2]
    window_w, window_h = window_size
    step_x = int(window_w * (1 - overlap_ratio))
    step_y = int(window_h * (1 - overlap_ratio))

    windows = []
    x = 0
    while x < w:
        y = 0
        while y < h:
            x2 = x + window_w
            y2 = y + window_h
            if x2 > w:
                x2 = w
                x = x2 - window_w
            if y2 > h:
                y2 = h
                y = y2 - window_h
            window = img[y:y2, x:x2]
            windows.append(((x, y, x2, y2), window))
            y += step_y
            if y + window_h > h: break
        x += step_x
        if x + window_w > w: break

    for i, (window_coord, window_image) in enumerate(windows):
        output_image_name = f"{image_name}_window{i}{image_ext}"
        output_annotation_name = f"{image_name}_window{i}.txt"
        
        output_image_path = output_img_dir / output_image_name
        cv2.imwrite(str(output_image_path), window_image)

        new_annotations = []
        window_x1, window_y1, window_x2, window_y2 = window_coord
        
        for ann in annotations:
            class_id = ann[0]
            polygon = list(map(float, ann[1:]))
            
            abs_polygon = []
            for j in range(0, len(polygon), 2):
                px = polygon[j] * img.shape[1]
                py = polygon[j+1] * img.shape[0]
                abs_polygon.extend([px, py])
                
            old_max_x = max(abs_polygon[j] for j in range(0, len(abs_polygon), 2))
            old_max_y = max(abs_polygon[j+1] for j in range(0, len(abs_polygon), 2))
            old_min_x = min(abs_polygon[j] for j in range(0, len(abs_polygon), 2))
            old_min_y = min(abs_polygon[j+1] for j in range(0, len(abs_polygon), 2))
            
            in_window = False
            in_side = False
            for j in range(0, len(abs_polygon), 2):
                px = abs_polygon[j]
                py = abs_polygon[j+1]
                if old_max_x >= window_x2 or old_min_x <= window_x1 or old_min_y <= window_y1 or old_max_y >= window_y2:
                    in_window = False
                    in_side = True
                    break
                if window_x1 <= px <= window_x2 and window_y1 <= py <= window_y2 and not in_side:
                    in_window = True
                    break
            
            if not in_window:
                continue
                
            cropped_polygon = []
            for j in range(0, len(abs_polygon), 2):
                px = abs_polygon[j]
                py = abs_polygon[j+1]
                new_x = min(max(px - window_x1, 0), window_w)
                new_y = min(max(py - window_y1, 0), window_h)
                cropped_polygon.extend([new_x, new_y])
                
            new_polygon = []
            for j in range(0, len(cropped_polygon), 2):
                new_polygon.extend([cropped_polygon[j] / window_w, cropped_polygon[j+1] / window_h])
                
            new_annotations.append([class_id] + new_polygon)

        output_annotation_path = output_ann_dir / output_annotation_name
        with open(output_annotation_path, 'w') as f:
            for ann in new_annotations:
                f.write(' '.join(map(str, ann)) + '\n')
                
    print(f"✅ 处理完成: {image_name} 切分为 {len(windows)} 个窗口")

# ============================================================================
# FORMAT CONVERSION (实战增强)
# ============================================================================

def convert_txt_to_json(txt_path, save_path, labels_list, img_width, img_height):
    """
    [实战经验] 将 YOLO txt 格式标签反向转换为 JSON 格式 (如 ISAT/LabelMe 格式)
    用于将模型的预测结果转换为可视化标注工具可读取的格式。
    
    Args:
        txt_path: YOLO txt 标签目录
        save_path: JSON 保存目录
        labels_list: 类别名称列表 (如 ['person', 'car'])
        img_width: 原始图像宽度
        img_height: 原始图像高度
    """
    import os
    import json
    import glob
    from pathlib import Path
    from tqdm import tqdm
    
    os.makedirs(save_path, exist_ok=True)
    search_path = os.path.join(txt_path, '*.txt')
    file_list = glob.glob(search_path, recursive=True)
    
    print(f"开始转换: 找到 {len(file_list)} 个 txt 文件")
    
    for txt_item in tqdm(file_list, desc="TXT to JSON"):
        lines = []
        with open(txt_item, 'r') as file:
            for line in file:
                lines.append(line.strip().split())
                
        # 基础 JSON 结构 (以 ISAT/LabelMe 为参考)
        transformed_annotation = {
            'version': '5.3.1',
            'flags': {},
            'shapes': [],
            'imagePath': Path(txt_item).stem + ".jpg",
            'imageData': None,
            'imageHeight': img_height,
            'imageWidth': img_width
        }
        
        # 处理每一行预测框 (反归一化)
        for line_item in lines:
            if len(line_item) < 5:
                continue
                
            class_id = int(line_item[0])
            line_label = labels_list[class_id] if class_id < len(labels_list) else str(class_id)
            
            # 中心点和宽高 (归一化值)
            dx = float(line_item[1])
            dy = float(line_item[2])
            dw = float(line_item[3])
            dh = float(line_item[4])
            
            # 反归一化计算绝对坐标
            x = dx * img_width
            y = dy * img_height
            w = dw * img_width
            h = dh * img_height
            
            # 计算左上角和右下角
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            transformed_shape = {
                'label': line_label,
                'points': [[float(x1), float(y1)], [float(x2), float(y2)]],
                'group_id': None,
                'description': '',
                'shape_type': 'rectangle',
                'flags': {}
            }
            transformed_annotation['shapes'].append(transformed_shape)
            
        output_annotation_path = Path(save_path) / (Path(txt_item).stem + '.json')
        with open(output_annotation_path, 'w') as f:
            json.dump(transformed_annotation, f, indent=2)
            
    print(f"✅ 成功将 {len(file_list)} 个文件转换为 JSON")

def coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO format to YOLO format
    
    Args:
        coco_json_path: Path to COCO JSON annotation file
        output_dir: Output directory for YOLO format
    """
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_dir = Path(output_dir) / 'images'
    labels_dir = Path(output_dir) / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Map category IDs to YOLO class indices
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    # Process each image
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        img_name = img_info['file_name']
        
        # Find annotations for this image
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # Create label file
        label_file = labels_dir / f"{Path(img_name).stem}.txt"
        with open(label_file, 'w') as f:
            for ann in img_annotations:
                # Get bounding box [x, y, width, height] in COCO format
                bbox = ann['bbox']
                x_min, y_min, width, height = bbox
                
                # Convert to YOLO format
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                
                # Get class ID
                class_id = categories.get(ann['category_id'], 0)
                
                # Write to file
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    print(f"Conversion complete. Output directory: {output_dir}")
    return output_dir

def voc_to_yolo(voc_xml_path, output_dir, class_mapping):
    """
    Convert VOC XML format to YOLO format
    
    Args:
        voc_xml_path: Path to VOC XML annotation file
        output_dir: Output directory for YOLO format
        class_mapping: Dictionary mapping class names to YOLO class IDs
    """
    
    # Parse XML
    tree = ET.parse(voc_xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Create output file
    output_file = Path(output_dir) / f"{Path(voc_xml_path).stem}.txt"
    
    with open(output_file, 'w') as f:
        # Process each object
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text
            class_id = class_mapping.get(class_name, 0)
            
            # Get bounding box
            bndbox = obj.find('bndbox')
            x_min = float(bndbox.find('xmin').text)
            y_min = float(bndbox.find('ymin').text)
            x_max = float(bndbox.find('xmax').text)
            y_max = float(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Conversion complete. Output file: {output_file}")
    return output_file

# ============================================================================
# DATASET SPLITTING TOOLS
# ============================================================================

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """
    将数据集划分为训练集和验证集 (实战增强版)
    
    Args:
        image_dir: 包含所有图像的目录
        label_dir: 包含所有标签文件的目录
        output_dir: 划分后的输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    import os
    import shutil
    import random
    from pathlib import Path
    from tqdm import tqdm
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # 支持的图像格式
    img_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in img_formats]
    
    # 随机打乱
    random.shuffle(images)
    
    # 计算划分索引
    num_images = len(images)
    num_train = int(num_images * train_ratio)
    
    train_images = images[:num_train]
    val_images = images[num_train:]
    
    # 创建目录结构
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    print(f"开始划分数据集: 总计 {num_images} 张图片")
    print(f"训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")
    
    def copy_files(img_list, split_name):
        missing_labels = 0
        for img_name in tqdm(img_list, desc=f"处理 {split_name} 集"):
            img_path = image_dir / img_name
            label_name = Path(img_name).stem + '.txt'
            label_path = label_dir / label_name
            
            # 复制图像
            shutil.copy(img_path, output_dir / 'images' / split_name / img_name)
            
            # 检查并复制标签
            if label_path.exists():
                shutil.copy(label_path, output_dir / 'labels' / split_name / label_name)
            else:
                missing_labels += 1
                
        if missing_labels > 0:
            print(f"\u26a0\ufe0f 警告: {split_name} 集中有 {missing_labels} 张图片缺少对应的标签文件")

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    print("\u2705 数据集划分完成！")

# ============================================================================
# DATASET ANALYSIS TOOLS
# ============================================================================

def analyze_dataset(labels_dir):
    """
    Analyze dataset statistics
    
    Args:
        labels_dir: Directory containing label files
    
    Returns:
        Dictionary with dataset statistics
    """
    
    from collections import Counter
    
    label_files = list(Path(labels_dir).glob('*.txt'))
    
    # Count objects per class
    class_counts = Counter()
    bbox_stats = {'widths': [], 'heights': []}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    
                    # Bbox dimensions
                    width = float(parts[3])
                    height = float(parts[4])
                    bbox_stats['widths'].append(width)
                    bbox_stats['heights'].append(height)
    
    # Calculate statistics
    total_files = len(label_files)
    total_objects = sum(class_counts.values())
    
    # Print statistics
    print(f"Total label files: {total_files}")
    print(f"Total objects: {total_objects}")
    print("\nClass distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} objects ({count/total_objects*100:.1f}%)")
    
    # Try to plot distribution if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        if bbox_stats['widths'] and bbox_stats['heights']:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.hist(bbox_stats['widths'], bins=50, alpha=0.7)
            plt.title('Bounding Box Width Distribution')
            plt.xlabel('Normalized Width')
            
            plt.subplot(1, 2, 2)
            plt.hist(bbox_stats['heights'], bins=50, alpha=0.7)
            plt.title('Bounding Box Height Distribution')
            plt.xlabel('Normalized Height')
            
            plt.tight_layout()
            plt.show()
        else:
            print("No bounding boxes found to plot.")
    except ImportError:
        print("Note: matplotlib not installed. Install with: pip install matplotlib")
        print("Skipping visualization.")
    
    # Return statistics
    return {
        'total_files': total_files,
        'total_objects': total_objects,
        'class_counts': dict(class_counts),
        'bbox_stats': bbox_stats,
    }
    
    return {
        'total_files': len(label_files),
        'total_objects': sum(class_counts.values()),
        'class_distribution': dict(class_counts),
        'bbox_stats': bbox_stats
    }

# ============================================================================
# DATASET CONFIGURATION TOOLS
# ============================================================================

def create_data_yaml(dataset_path, class_names, output_path='data.yaml'):
    """
    Create YOLO dataset configuration file
    
    Args:
        dataset_path: Path to dataset root directory
        class_names: List of class names or dictionary mapping class IDs to names
        output_path: Output path for data.yaml file
    """
    
    # Try to import yaml, fall back to JSON or print error
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
        print("Warning: PyYAML not installed. Cannot create YAML file.")
        print("Install with: pip install pyyaml")
        return None
    
    # Convert class_names to dictionary if it's a list
    if isinstance(class_names, list):
        class_dict = {i: name for i, name in enumerate(class_names)}
    else:
        class_dict = class_names
    
    # Create data.yaml content
    data = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': class_dict,
        'nc': len(class_dict),
    }
    
    # Write to file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Dataset configuration created: {output_path}")
    return output_path

def validate_dataset(dataset_path, labels_dir=None):
    """
    Validate YOLO dataset structure and files
    
    Args:
        dataset_path: Path to dataset root directory OR images directory
        labels_dir: Optional path to labels directory (if dataset_path is images directory)
    
    Returns:
        True if validation passed, False otherwise
    """
    
    dataset_path = Path(dataset_path)
    issues = []
    
    if labels_dir is None:
        # Assume dataset_path is the root directory with YOLO structure
        # Check directory structure
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        missing_dirs = []
        
        for dir_path in required_dirs:
            if not (dataset_path / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"Missing directories: {missing_dirs}")
            return False
        
        # Check for corresponding label files
        for split in ['train', 'val']:
            image_dir = dataset_path / 'images' / split
            label_dir = dataset_path / 'labels' / split
            
            if not image_dir.exists() or not label_dir.exists():
                continue
            
            # Get all images
            images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
            
            for img_file in images:
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    issues.append(f"Missing label for {img_file.relative_to(dataset_path)}")
                
                # Validate label file format
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                issues.append(f"Invalid format in {label_file.relative_to(dataset_path)} line {line_num}")
                                continue
                            
                            # Check values are within [0, 1]
                            try:
                                values = list(map(float, parts[1:]))
                                if any(v < 0 or v > 1 for v in values):
                                    issues.append(f"Values out of range in {label_file.relative_to(dataset_path)} line {line_num}")
                            except ValueError:
                                issues.append(f"Non-numeric values in {label_file.relative_to(dataset_path)} line {line_num}")
    else:
        # dataset_path is images directory, labels_dir is labels directory
        labels_dir = Path(labels_dir)
        
        if not dataset_path.exists():
            issues.append(f"Images directory does not exist: {dataset_path}")
        
        if not labels_dir.exists():
            issues.append(f"Labels directory does not exist: {labels_dir}")
        
        if dataset_path.exists() and labels_dir.exists():
            # Get all images
            images = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
            
            for img_file in images:
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    issues.append(f"Missing label for {img_file.name}")
                
                # Validate label file format
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                issues.append(f"Invalid format in {label_file.name} line {line_num}")
                                continue
                            
                            # Check values are within [0, 1]
                            try:
                                values = list(map(float, parts[1:]))
                                if any(v < 0 or v > 1 for v in values):
                                    issues.append(f"Values out of range in {label_file.name} line {line_num}")
                            except ValueError:
                                issues.append(f"Non-numeric values in {label_file.name} line {line_num}")
    
    if issues:
        print(f"Validation issues found ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    
    print("Dataset validation passed!")
    return True

# ============================================================================
# AUGMENTATION TOOLS
# ============================================================================

def get_augmentation_pipeline(mode='train'):
    """
    Get augmentation pipeline for training or validation
    
    Args:
        mode: 'train' for training augmentations, 'val' or 'test' for validation/test
    
    Returns:
        Albumentations pipeline or None if albumentations not installed
    """
    
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        if mode == 'train':
            return A.Compose([
                A.Resize(640, 640),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.HueSaturationValue(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.Blur(blur_limit=3, p=0.1),
                A.CLAHE(p=0.1),
                A.ToGray(p=0.1),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        elif mode in ['val', 'test', 'validation']:
            return A.Compose([
                A.Resize(640, 640),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        else:
            print(f"Warning: Invalid mode '{mode}'. Using validation pipeline.")
            return A.Compose([
                A.Resize(640, 640),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    except ImportError:
        print("Warning: albumentations not installed. Augmentation pipeline not available.")
        print("Install with: pip install albumentations")
        return None

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("YOLO Dataset Tools")
    print("=" * 60)
    
    # Example: Create dataset configuration
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane']
    config_path = create_data_yaml('./my_dataset', class_names)
    print(f"Created dataset config: {config_path}")
    
    # Example: Analyze dataset
    print("\nTo analyze a dataset:")
    print("  from dataset_tools import analyze_dataset")
    print("  stats = analyze_dataset('./my_dataset/labels/train')")
    
    # Example: Split dataset
    print("\nTo split a dataset:")
    print("  from dataset_tools import split_dataset")
    print("  splits = split_dataset('./images', './labels', './split_dataset')")
    
    print("\nAvailable functions:")
    print("  - coco_to_yolo(): Convert COCO format to YOLO format")
    print("  - voc_to_yolo(): Convert VOC format to YOLO format")
    print("  - split_dataset(): Split dataset into train/val/test")
    print("  - analyze_dataset(): Analyze dataset statistics")
    print("  - create_data_yaml(): Create dataset configuration")
    print("  - validate_dataset(): Validate dataset structure (YOLO format or separate dirs)")
    print("  - get_augmentation_pipeline(): Get augmentation pipeline")