import os
import csv

label_list = ["306"]

# 定义函数来读取标签文件
def read_labels(label_dir):
    labels = {}
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            img_name = filename.replace(".txt", ".jpg")  # 假设图片是.jpg格式
            with open(os.path.join(label_dir, filename), "r") as file:
                lines = file.readlines()
                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:  # 真实标签文件
                        category, x, y, w, h = parts
                        boxes.append(f"{category} {x} {y} {w} {h}")
                labels[img_name] = boxes
    return labels

# 定义函数来读取预测标签文件
def read_predictions(pred_dir):
    predictions = {}
    for filename in os.listdir(pred_dir):
        if filename.endswith(".txt"):
            img_name = filename.replace(".txt", ".jpg")  # 假设图片是.jpg格式
            with open(os.path.join(pred_dir, filename), "r") as file:
                lines = file.readlines()
                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 6:  # 预测标签文件
                        category, x, y, w, h, confidence = parts
                        boxes.append(f"{category} {x} {y} {w} {h} {confidence}")
                predictions[img_name] = boxes
    return predictions

# 定义函数来生成CSV文件
def generate_csv(label_dir, pred_dir, output_csv):
    # 读取标签文件和预测文件
    true_labels = read_labels(label_dir)
    predicted_labels = read_predictions(pred_dir)

    # 打开CSV文件，写入数据
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "True Labels", "Predicted Labels", "Confidence"])

        # 遍历真实标签和预测标签
        for img_name in true_labels:
            if img_name in predicted_labels:
                true_labels_str = label_list[int(true_labels[img_name][0].split(" ")[0])]
                pred_labels_str = label_list[int(predicted_labels[img_name][0].split(" ")[0])]
                confidence = " ".join([item.split()[-1] for item in predicted_labels[img_name]])
                writer.writerow([img_name, true_labels_str, pred_labels_str, confidence])

# 设置标签文件夹和预测文件夹路径
label_dir = "/home/zhangx/dataset/B9/20250106_B_dataset/val"  # 真实标签文件夹路径
pred_dir = "runs/detect/predict3/labels"  # 预测标签文件夹路径
output_csv = "output.csv"  # 输出CSV文件路径

# 调用函数生成CSV
generate_csv(label_dir, pred_dir, output_csv)

print(f"CSV file saved to {output_csv}")
