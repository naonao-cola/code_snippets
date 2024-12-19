import shutil

from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np


def prettyXml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if element.text == None or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
    # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        prettyXml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def creat_xml(boxes, codes, img_path, xml_path, width, height):
    if codes[0] != 'OK':
        shutil.copy(xml_path, os.path.abspath(os.path.join(img_path, '../')))
        old_name = os.path.abspath(os.path.join(img_path, '../')) + f'/{xml_path.split("/")[-1]}'

        xml_file = old_name
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root.find('filename').text = img_path.split('/')[-1]
        path = "\\".join(root.find('path').text.split("\\")[:-3]) + '\\' + '\\'.join(img_path.split('/')[-3:])
        root.find('path').text = path

        root.find('size').find('width').text = width
        root.find('size').find('height').text = height
        for obj in root.findall('object'):
            root.remove(obj)
        for i, code in enumerate(codes):
            # for row in boxes[i]:
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            pose = ET.SubElement(obj, 'pose')
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            name.text = code
            pose.text = 'Unspecified'
            xmin.text = str(int(boxes[i][0]))
            ymin.text = str(int(boxes[i][1]))
            xmax.text = str(int(boxes[i][2]))
            ymax.text = str(int(boxes[i][-1]))

            prettyXml(root, '    ', '\n')  # 执行美化方法
            tree.write(xml_file)
        if img_path.endswith('jpg'):
            new_name = img_path.replace('jpg', 'xml')
        else:
            new_name = img_path.replace('bmp', 'xml')

        os.rename(old_name, new_name)

