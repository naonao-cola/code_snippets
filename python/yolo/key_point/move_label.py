
import os
import shutil

# move json

# 将训练的标注文件移动到训练目录，验证的标注文件手动复制
val_xml_path = r'/data/proj/www/repo/yolo8_test/dataset/pin/coco2yolo'
img_path = r'/data/proj/www/repo/yolo8_test/dataset/pin/images'
move_path = r'/data/proj/www/repo/yolo8_test/dataset/pin/images'
for filename in os.listdir(img_path):
    for filename2 in os.listdir(val_xml_path):
        img_list = filename.split('.')
        label_list = filename2.split('.')
        if len(img_list) == 2:
            img_name = ''
            label_name = ''
            for i in range(len(img_list)-1):
                img_name = img_name+img_list[i]
            for i in range(len(label_list)-1):
                label_name = label_name+label_list[i]
            if img_name == label_name:
                shutil.move(val_xml_path + '/' + filename2, move_path)
        else:
            if filename.split('.')[0] == filename2.split('.')[0]:
                shutil.move(val_xml_path + '/' + filename, move_path)
