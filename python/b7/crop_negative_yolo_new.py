### crop_negative_yolo_new.py

'''
1、缺陷中心裁剪
2、遍历标注框，调整标注框
3、保存小图
4、保存小图对应的txt标注文件
'''
import glob
import xml.etree.ElementTree as ET
import xml

import os
join = os.path.join

from PIL import Image, ImageOps
from tqdm import tqdm

class gen_xml_file():
    def __init__(self):
        self.annotation

    def create_file_head(self, xml_folder, xml_filename, shape):
        self.annotation = ET.Element("annotation")
        # 添加子元素        
        folder = ET.SubElement(self.annotation, "folder")
        folder.text = xml_folder

        filename = ET.SubElement(self.annotation, "filename")
        filename.text = xml_filename

        segmented = ET.SubElement(self.annotation, "segmented")
        segmented.text = str(0)

        source = ET.SubElement(self.annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = 'Unknown'

        size = ET.SubElement(self.annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(shape[1])
        height = ET.SubElement(size, "height")
        height.text = str(shape[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(shape[2])
        return self.annotation

    def write_info2xml(self, m_label, temp_box):
        # 填充标注信息
        label = m_label
        m_name = label[0:-1]
        m_difficult = label[-1]
        
        m_subConf = 1
        m_xmin = temp_box[0]
        m_ymin = temp_box[1]
        m_xmax = temp_box[2]
        m_ymax = temp_box[3]

        m_object = ET.SubElement(self.annotation, "object")
        name = ET.SubElement(m_object, "name")
        name.text = m_name
        pose = ET.SubElement(m_object, "pose")
        pose.text = 'Unspecifed'
        truncated = ET.SubElement(m_object, "truncated")
        truncated.text = str(0)
        difficult = ET.SubElement(m_object, "difficult")
        difficult.text = str(m_difficult)
        contrast = ET.SubElement(m_object, "contrast")
        contrast.text = str(0)
        luminance = ET.SubElement(m_object, "luminance")
        luminance.text = str(0)
        subConf = ET.SubElement(m_object, "subConf")
        subConf.text = str(m_subConf)
        bndbox = ET.SubElement(m_object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(m_xmin))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(m_ymin))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(m_xmax))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(m_ymax))

    def finish_file_tail(self, concat_save_path, xml_filename):
        # 将 XML 结构保存为文件
        save_path = concat_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        xml_filepath = f"{save_path}\\{xml_filename}.xml"
        
        tree = ET.ElementTree(self.annotation)
        tree.write(xml_filepath, encoding="utf-8", xml_declaration=True)

        # 使用 xml.dom.minidom 格式化 XML 文件
        dom = xml.dom.minidom.parse(xml_filepath)
        with open(xml_filepath, "w", encoding="utf-8") as f:
            f.write(dom.toprettyxml(indent="    "))  # 使用四个空格作为缩进


class annotation_xml2yolo:
    def __init__(self, labels, save_path) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.labels = labels
        # 保存labels文件
        labels_file = join(save_path, 'labels.txt')
        classes_file = join(save_path, 'classes.txt')

        with open(labels_file, 'w', encoding='utf-8') as l_f:
            for k in range(len(labels)):
                data = str(k) + ' ' + labels[k] + '\n'
                l_f.write(data)

        with open(classes_file, 'w', encoding='utf-8') as c_f:
            for k in range(len(labels)):
                data = labels[k] + '\n'
                c_f.write(data)

    def xml2txt(self, m_label, temp_box, width, height, f):
        # 获取label的id
        index = self.labels.index(m_label)
        # yolo标注信息为：中心点、宽高
        cen_x = (temp_box[0] + temp_box[2]) / 2.0
        cen_y = (temp_box[1] + temp_box[3]) / 2.0
        w = temp_box[2] - temp_box[0]
        h = temp_box[3] - temp_box[1]

        normal_cen_x = round(cen_x / width, 4)
        normal_cen_y = round(cen_y / height, 4)
        normal_w = round(w / width, 4)
        normal_h = round(h / height, 4)

        # data = index + ' ' + normal_lt_x + ' ' + normal_lt_y + ' ' + normal_rb_x + ' ' + normal_rb_y + '\n'
        data = f"{index} {normal_cen_x} {normal_cen_y} {normal_w} {normal_h}\n"

        f.write(data)

    def adjust_box(self, i, box, tolerance, defect_save_path, file_path):
        # 生成xml文件
        m_xml = gen_xml_file()
        xml_folder = defect_save_path
        xml_filename = file_path.spilt("\\")[-1]
        shape = [box[2] - box[0], box[3] - box[1], 3]
        m_xml.create_file_head(xml_folder, xml_filename, shape)
        
        # 生成txt文件
        txt_f =  open(file_path + ".txt", 'w', encoding='utf-8')


        # 遍历缺陷 、、写新标注
        for j in range(len(self.name_list)):
            m_label = self.name_list[j]
            
            # 图像宽高
            width = box[2] - box[0]
            height = box[3] - box[1]

            # 标注框宽高
            label_width = self.x_max_list[j] - self.x_min_list[j]
            label_height = self.y_max_list[j] - self.y_min_list[j]

            max_data = max(label_width, label_height)
            min_data = min(label_width, label_height)
            ratio = max_data / min_data

            # 对于小目标缺陷，需要确保标注完整
            if ratio < 5 and max_data < 40:
                lt_x = self.x_min_list[j]
                lt_y = self.y_min_list[j]
                rb_x = self.x_max_list[j]
                rb_y = self.y_max_list[j]

                if lt_x >= box[0] and lt_y >= box[1] and rb_x <=box[2] and rb_y <= box[3]:
                    temp_box = (lt_x, lt_y, rb_x, rb_y)
                    # xml标注转yolo标注
                    print("正在执行xml标注转yolo标注")
                    self.xml2txt(m_label, temp_box, width, height, txt_f)
                    m_xml.write_info2xml(m_label, temp_box)
                
            else:
                cen_x = int((self.x_min_list[j] + self.x_max_list[j]) / 2.0)
                cen_y = int((self.y_min_list[j] + self.y_max_list[j]) / 2.0)

                # 中心点容忍框
                lt_x = cen_x - tolerance / 2
                lt_y = cen_y - tolerance / 2
                rb_x = lt_x + tolerance
                rb_y = lt_y + tolerance

                # 判断标注是否在小图内部
                if (lt_x >= box[0] and lt_x <= box[2] and lt_y >= box[1] and lt_y <= box[3]) or \
                    (lt_x >= box[0] and lt_x <= box[2] and rb_y >= box[1] and rb_y <= box[3]) or \
                    (rb_x >= box[0] and rb_x <= box[2] and lt_y >= box[1] and lt_y <= box[3]) or \
                    (rb_x >= box[0] and rb_x <= box[2] and rb_y >= box[1] and rb_y <= box[3]):

                    # 调整标注
                    lt_x = max(self.x_min_list[j] - box[0], 0)
                    lt_y = max(self.y_min_list[j] - box[1], 0)
                    rb_x = min(self.x_max_list[j] - box[0], width)
                    rb_y = min(self.y_max_list[j] - box[1], height)
                    
                    temp_box = (lt_x, lt_y, rb_x, rb_y)
                    # xml标注转yolo标注
                    print("正在执行xml标注转yolo标注")
                    self.xml2txt(m_label, temp_box, width, height, txt_f)
                    m_xml.write_info2xml(m_label, temp_box)

        m_xml.finish_file_tail(xml_folder, xml_filename)
        txt_f.close()



    def img2xml(self, img_file, save_path, model_size, tolerance):
        self.name_list = []
        self.x_min_list = []
        self.y_min_list = []
        self.x_max_list = []
        self.y_max_list = []

        # 读取图像对应的xml标注文件
        # xml_path = img_file.replace('jpg', 'xml')
        xml_path = img_file.replace('bmp', 'xml')
        # 解析XML
        try:
            mytree = ET.parse(xml_path)
        except:
            return -1
        myroot = mytree.getroot()

        # 存储整体的标注信息
        self.img_width = -1
        self.img_height = -1

        # 获取图像宽高
        self.img_width = int(myroot.find('size').find('width').text)
        self.img_height = int(myroot.find('size').find('height').text)

        # 获取标注object信息
        for obj in myroot.iter('object'):
            # label名称：需要区分等级的Label要增加等级
            difficult = obj.find('difficult').text
            name = obj.find('name').text
            label = f"{name}{str(difficult)}"
            self.name_list.append(label)

            # box
            box = obj.find('bndbox')
            self.x_min_list.append(float(box.find('xmin').text))
            self.y_min_list.append(float(box.find('ymin').text))
            self.x_max_list.append(float(box.find('xmax').text))
            self.y_max_list.append(float(box.find('ymax').text)) 
        
        if len(self.name_list) < 1:
            return -1
        
        # 截图和调整标注box        
        img = Image.open(img_file)
        img = ImageOps.exif_transpose(img)
        for i in range(len(self.name_list)):
            cen_x = int((self.x_min_list[i] + self.x_max_list[i]) / 2.0)
            cen_y = int((self.y_min_list[i] + self.y_max_list[i]) / 2.0)

            lt_x = cen_x - model_size / 2
            lt_y = cen_y - model_size / 2
            rb_x = lt_x + model_size
            rb_y = lt_y + model_size

            # 判断是否超界
            if lt_x < 0 :
                over_size = 0 - lt_x
                lt_x = 0
                rb_x = rb_x + over_size

            if lt_y < 0 :
                over_size = 0 - lt_y
                lt_y = 0
                rb_y = rb_y + over_size

            if rb_x > self.img_width:
                over_size = rb_x - self.img_width
                rb_x = self.img_width
                lt_x = lt_x - over_size

            if rb_y > self.img_height:
                over_size = rb_y - self.img_height
                rb_y = self.img_height
                lt_y = lt_y - over_size

            # 存缺陷小图：根据Label区分文件夹
            defect_save_path = join(save_path, self.name_list[i])            
            if not os.path.exists(defect_save_path):
                os.makedirs(defect_save_path)
                print("创建Label文件夹：", defect_save_path)

            box = (lt_x, lt_y, rb_x, rb_y)
            small_img = img.crop(box)
            img_name = img_file.split('\\')[-1].split('.')[0]
            simg_path = join(defect_save_path, img_name + '_' + str(i) + '.bmp')
            small_img.save(simg_path)

            # 存储标注
            file_path = join(defect_save_path, img_name + '_' + str(i))
            
            # 调整标注框
            print("正在执行生成标注文件")
            self.adjust_box(i, box, tolerance, defect_save_path, file_path)


if __name__ == "__main__":
    model_size = 640
    tolerance = 10

    source_path = r'D:\0_work\dataset_ijp_augment\negative'
    save_path = r'D:\0_work\dataset_ijp_augment\negative_crop'
    # 如果缺陷Code有等级，需要区分等级，如：STM1,STM2
    labels = ['ESD1', 'ESD9', 'STM0', 'STM1', 'STM2', 'Line0', 'Line1', 'Line2', 'Line3']

    # 遍历文件夹
    for file in os.listdir(source_path):
        print('当前处理文件：', file)
    
        source_image = join(source_path, file)
        save_image = join(save_path, file)

        img_files = glob.glob(source_image + '/**/*.bmp', recursive=True)
        m_class = annotation_xml2yolo(labels, save_image)
        for img_file in tqdm(img_files):
            m_class.img2xml(img_file, save_image, model_size, tolerance)
