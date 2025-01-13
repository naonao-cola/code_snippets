'''
yolo标注和xml标注互转
'''

import xml.etree.ElementTree as ET
import xml.dom.minidom
import os

join = os.path.join
from tqdm import tqdm
import glob
import random
import xml.etree.ElementTree as ET


def xml2yolo(xml_file, code_list):
    '''
    code_list 的顺序必须和标注文件中的一致
    '''

    mytree = ET.parse(xml_file)
    myroot = mytree.getroot()

    # 获取图像宽高
    img_width = int(myroot.find('size').find('width').text)
    img_height = int(myroot.find('size').find('height').text)

    # yolo存储文件
    yolo_file = xml_file.replace('.xml', '.txt')
    f = open(yolo_file, 'w', encoding='utf-8')

    # 获取标注object信息
    for obj in myroot.iter('object'):
        # label名称：需要区分等级的Label要增加等级
        name = obj.find('name').text
        label = f"{name}"
        # box
        box = obj.find('bndbox')
        xmin = float(box.find('xmin').text)
        ymin = float(box.find('ymin').text)
        xmax = float(box.find('xmax').text)
        ymax = float(box.find('ymax').text)

        # 获取label的id
        try:
            index = code_list.index(label)
        except:
            index = code_list.index(label)

        # yolo标注信息为：中心点、宽高
        cen_x = (xmin + xmax) / 2.0
        cen_y = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin

        normal_cen_x = round(cen_x / img_width, 4)
        normal_cen_y = round(cen_y / img_height, 4)
        normal_w = round(w / img_width, 4)
        normal_h = round(h / img_height, 4)

        data = f"{index} {normal_cen_x} {normal_cen_y} {normal_w} {normal_h}\n"

        f.write(data)
    f.close()


def is_empty(file):
    return os.stat(file).st_size == 0


def data_split(txt_path, fname, ratio, time):
    length = len(fname)  # 获取数据集的总长度
    train_num = int(length * ratio[0])

    txt_path = os.path.dirname(txt_path) + f'/{time}'
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    train_data = open(txt_path + '/train.txt', 'w')
    val_data = open(txt_path + '/val.txt', 'w')

    val_idx = random.sample(range(length), train_num)  # 得到按比例拆分后的元素下标

    index = [False for _ in range(length)]  # 记录所有列表中元素是否被选中（可以通过这种方式来减少时间复杂度）
    for i in val_idx:
        index[i] = True

    # 开始抽取
    for i, element in enumerate(index):
        if not is_empty(fname[i].replace('.jpg', '.txt')):
            if element == False:
                val_data.writelines(fname[i] + '\n')
            else:
                train_data.writelines(fname[i] + '\n')
        else:
            continue

    train_data.close()
    val_data.close()


def main_xml2yolo(save_path, code_list):
    xml_filePaths = glob.glob(save_path + '/**/*.xml', recursive=True)
    for xml_file in tqdm(xml_filePaths):
        xml2yolo(xml_file, code_list)
    print(code_list)


if __name__ == '__main__':
    time = 'train_val/01113'
    ratio = (0.8, 0.2) 
    data_path = '/home/tvt/hjx/project/B19/data/TVB/train_data'
    # 标注code路径
    # TVPS
    # code_list = ["P302", "P303", "P304", "P304_OK", "P306", "P309", "P501", "P504", "P510", "B501", "B510A", "B510B", "P705", "repair", "ECHO", "P702"]
    # POAPS
    # code_list = ["P702", "P501", "P306", "P304", "P303", "repair", "P705", "P510", "TFT"]
    # TVB
    code_list = ["B302", "B303", "B304", "B304_OK", "B306", "B309", "B501", "B504", "B510A", "B510B", "B705", "repair", "ECHO", "B702"]

    # ------------------xml转yolo标注--------------------#
    main_xml2yolo(data_path, code_list)

    # ------------------生成数据集--------------------#
    total_list = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith('.txt') and file.split('.')[0] not in ['classes', 'labels']:
                img_path = file.replace('.txt', '.jpg')
                img_path = os.path.join(root, img_path)
                total_list.append(img_path)

    data_split(data_path, total_list, ratio, time)