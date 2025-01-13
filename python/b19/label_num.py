import os
import xml.etree.ElementTree as ET
from collections import Counter
import matplotlib.pyplot as plt

# 定义 XML 文件所在的目录
xml_dir = '/home/hjx/project/B19/data/TVB/train_data'

# 初始化类别计数器
category_counts = {}

# 遍历所有的 XML 文件
def count_categories(xml_dir):
    category_counts = Counter()
    for root_dir, _, files in os.walk(xml_dir):
        for xml_file in files:
            if xml_file.endswith('xml'):
                tree = ET.parse(os.path.join(root_dir, xml_file))
                root = tree.getroot()

                # 遍历每个 object 标签
                for obj in root.findall('object'):
                    category = obj.find('name').text
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1

    return category_counts

category_counts = count_categories(xml_dir)

# 准备绘图数据
categories = list(category_counts.keys())
counts = list(category_counts.values())

# 创建条形图
plt.figure(figsize=(12, 8))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Number of Instances')
plt.title('TVB Dataset')
plt.xticks(rotation=45, ha='right')  # 旋转标签以适应长名称
plt.tight_layout()  # 调整布局以防止标签被截断

# 显示图表
plt.show()

# 可选：保存图表到文件
plt.savefig('/home/hjx/project/B19/data/num.png', dpi=300)