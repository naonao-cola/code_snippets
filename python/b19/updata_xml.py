import glob
import os
import shutil
from collections import Counter

import xml.etree.ElementTree as ET


def updata(xml_path, step):
    code_list = []
    xml_files = glob.glob(xml_path + "/*/*.xml")
    if len(xml_files) == 0:
        xml_files = glob.glob(xml_path + "/*.xml")
    for xml_file in xml_files:
        img_file = xml_file.replace('xml', 'jpg')

        if not os.path.exists(img_file):
            os.remove(xml_file)
        else:
            tree = ET.parse(xml_file)
            myroot = tree.getroot()
            if not myroot.find('object'):
                os.remove(xml_file)
                os.remove(img_file)
                print(f'{img_file} removed!')
                continue
            else:
                for obj in myroot.iter('object'):
                    code = obj.find('name').text
                    if step == "TVB":
                        if code in ["P510", "P501"]:
                            os.remove(img_file)
                            os.remove(xml_file)
                            print("remove P510\P501")
                            break
                        elif code.startswith("P"):
                            code = "B" + code[1:]
                            obj.find('name').text = code
                            print("code name renamed")
                        else:
                            pass
                    if code not in code_list:
                        code_list.append(code)

                tree.write(xml_file)
    return code_list

def move_new_path(xml_path, save_path):
    # 根据标签第一个obj分到不同文件夹
    for root, _, files in os.walk(xml_path):
        for file in files:
            if file.endswith('.xml'):
                xml_file = os.path.join(root, file)
                img_file = xml_file.replace('xml', 'jpg')

                tree = ET.parse(xml_file)
                myroot = tree.getroot()
                if myroot.find('object'):
                    obj_1 = myroot.find('object')
                    name = obj_1.find('name').text
                    tar_path = os.path.join(save_path, name)
                    if not os.path.exists(tar_path):
                        os.makedirs(tar_path)
                    shutil.copy(xml_file, tar_path)
                    shutil.copy(img_file, tar_path)
                    print(f'{img_file} move finished!')
                tree.write(xml_file)


def count_categories(xml_dir):
    total = 0
    category_counts = Counter()
    for root_dir, _, files in os.walk(xml_dir):
        for i, xml_file in enumerate(files):
            if xml_file.endswith('xml'):
                tree = ET.parse(os.path.join(root_dir, xml_file))
                root = tree.getroot()
                total = total + 1
                # 遍历每个 object 标签
                for obj in root.findall('object'):
                    category = obj.find('name').text
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1

    category_counts['total'] = total
    print(category_counts)

if __name__ == "__main__":
    move = False
    step = "TVB"
    xml_path = '/home/tvt/hjx/project/B19/data/TVB/train_data'
    save_path = '/home/hjx/project/B19/data/TVPS/train_data'

    code_list = updata(xml_path, step)
    code_list = [x.replace("\'", "\"") for x in code_list]
    if move:
        move_new_path(xml_path, save_path)
    count_categories(xml_path)
    print(code_list)

