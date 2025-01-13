import albumentations as A
import cv2
import os
import tqdm 
from glob import glob

# 输入的图像和标签文件夹，以及输出文件夹路径
image_folder = "/home/tvt/zhangx/dataset/B19RPA/COAPS/20250109_COAPS_dataset/val"
label_folder = image_folder
output_folder = image_folder + "_aug"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 定义增强操作
transform = A.Compose([
    A.OneOf([
        A.GaussNoise(var_limit=(1.0, 5.0), p=0.9),  # 随机噪声
        A.MotionBlur(blur_limit=3, p=0.5) ], p=0.9),# 模糊
        A.VerticalFlip(p=0.5),  # 垂直翻转
        A.HorizontalFlip(p=0.5)
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

# 函数：读取 YOLO 格式标签
def read_yolo_labels(label_path):
    with open(label_path, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split()
            class_id = int(label_data[0])
            bbox = list(map(float, label_data[1:]))
            labels.append((class_id, bbox))
    return labels

# 函数：写入 YOLO 格式标签
def write_yolo_labels(output_label_path, augmented_bboxes, class_labels):
    with open(output_label_path, "w") as f:
        for bbox, class_id in zip(augmented_bboxes, class_labels):
            bbox_str = " ".join(map(str, [class_id, *bbox]))
            f.write(bbox_str + "\n")

# 获取图像路径
images = glob(os.path.join(image_folder, "*.jpg"))  # 假设图像格式为jpg
for img_path in tqdm.tqdm(images):
    # 读取图像和标签
    image = cv2.imread(img_path)
    img_name = os.path.basename(img_path).split(".")[0]
    label_path = os.path.join(label_folder, f"{img_name}.txt")
    
    # 如果标签文件不存在，则跳过该图像
    if not os.path.exists(label_path):
        print(f"Warning: Label for {img_name} not found.")
        continue
    
    # 读取标签并转换格式
    labels = read_yolo_labels(label_path)
    bboxes = [bbox for _, bbox in labels]
    class_labels = [class_id for class_id, _ in labels]
    

    for i in range(1):
        # 进行数据增强
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except:
            # print(img_path)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"remove {img_path}")
        augmented_image = augmented["image"]
        augmented_bboxes = augmented["bboxes"]
        
        # 保存增强后的图像和标签
        output_img_path = os.path.join(output_folder, f"{img_name}_aug_{i}.jpg")
        output_label_path = os.path.join(output_folder, f"{img_name}_aug_{i}.txt")
        
        # 保存增强图像
        cv2.imwrite(output_img_path, augmented_image)
        
        # 保存更新后的标签
        write_yolo_labels(output_label_path, augmented_bboxes, class_labels)

print("数据增强已完成，增强后的图片和标签文件保存在输出文件夹中。")
"test"
