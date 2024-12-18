'''
读取json结果文件，生成正确检出、误分类、过杀、漏检的excle文件
表头：图像名称、标注名、检出名、检出类型、置信度、标注图像、检出图像
'''
import xlsxwriter
from PIL import Image
import os
join = os.path.join
import glob
from tqdm import tqdm
import json
import random
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import time
img_ext = ".jpg"

class gen_excle_class:

    
    def __init__(self, report_path, code_path):
        ok_path = join(report_path, '正确检出.xlsx')
        overkill_path = join(report_path, '过检.xlsx')
        escape_path = join(report_path, '漏检.xlsx')
        misclass_path = join(report_path, '误分类.xlsx')
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        # 新建一个图像缓存文件temp,生成报告后可以删掉
        self.temp_path = join(report_path, 'temps')
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path) 
        self.label_images_path = join(report_path, 'labels')
        if not os.path.exists(self.label_images_path):
            os.makedirs(self.label_images_path)            
        self.pred_images_path = join(report_path, 'preds')
        if not os.path.exists(self.pred_images_path):
            os.makedirs(self.pred_images_path) 
        
        # 创建一个新的Excel工作簿
        self.ok_workbook = xlsxwriter.Workbook(ok_path)
        self.overkill_workbook = xlsxwriter.Workbook(overkill_path)
        self.escape_workbook = xlsxwriter.Workbook(escape_path)
        self.misclass_workbook = xlsxwriter.Workbook(misclass_path)

        # 定义格式以居中文本
        self.ok_cell_format = self.ok_workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        self.overkill_cell_format = self.overkill_workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        self.escape_cell_format = self.escape_workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        self.misclass_cell_format = self.misclass_workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # 添加一个工作表
        data = ['图像名称', '标注名称', '结果名称', '检出状态', '置信度', '标注图', '检出图']
        self.ok_worksheet = self.ok_workbook.add_worksheet("正确检出")
        self.ok_worksheet.set_default_row(100)
        self.add_row2write(self.ok_worksheet, 0, data, self.ok_cell_format)

        self.overkill_worksheet = self.overkill_workbook.add_worksheet("过检")
        self.overkill_worksheet.set_default_row(100)
        self.add_row2write(self.overkill_worksheet, 0, data, self.overkill_cell_format)
        
        self.escape_worksheet = self.escape_workbook.add_worksheet("漏检")
        self.escape_worksheet.set_default_row(100)
        self.add_row2write(self.escape_worksheet, 0, data, self.escape_cell_format)
        
        self.misclass_worksheet = self.misclass_workbook.add_worksheet("误分类")
        self.misclass_worksheet.set_default_row(100)   
        self.add_row2write(self.misclass_worksheet, 0, data, self.misclass_cell_format)

        # 获取code_list
        self.code_dict = {}
        self.color_dict = {}
        self.code_list = []
        print("code_path:", code_path)
        with open(code_path, 'r', encoding='utf-8') as f_txt:
            for line in f_txt:
                key = line.strip().split(' ')[0]
                value = line.strip().split(' ')[1]
                self.code_dict.update({key:value})
                self.code_list.append(value)
                print(line.strip(), key, value)
                # 生成随机的 RGB 颜色值
                red = int(line.strip().split(' ')[2])
                green = int(line.strip().split(' ')[3])
                blue = int(line.strip().split(' ')[4])
                self.color_dict.update({key:(blue, green, red)})

        # 行计数
        self.ok_row = 0
        self.overkill_row = 0
        self.escape_row = 0
        self.misclass_row = 0

    def add_row2write(self, worksheet, row, data, format):
        worksheet.set_column(5, 5, 40)
        worksheet.set_column(6, 6, 40)
        if row > 0:
            worksheet.set_row(row, 260)

        worksheet.write(row, 0, data[0], format)
        worksheet.write(row, 1, data[1], format)
        worksheet.write(row, 2, data[2], format)
        worksheet.write(row, 3, data[3], format)
        worksheet.write(row, 4, data[4], format)
        if data[5] == '标注图':
            worksheet.write(row, 5, data[5], format)
            worksheet.write(row, 6, data[6], format)
        else:
            worksheet.insert_image(row, 5, data[5], {'x_scale': 0.4, 'y_scale': 0.4})
            worksheet.insert_image(row, 6, data[6], {'x_scale': 0.4, 'y_scale': 0.4})

    def write_data2xlsx(self, image_name, result_status, pred_info, label_info):
        
        # 读图
        image_path = f"{self.label_images_path}/{image_name}"
        image_path = image_path.replace(".bmp", ".jpg")        
        src = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 结果图的路径
        pred_image_path = f"{self.pred_images_path}/{image_name}" 
        pred_image_path.replace('\\', '/')

        if result_status == 'overkill':
            # 绘制
            pred_box = pred_info[0:4]
            img_label_path = self.draw_circle_label(src, pred_box, image_name)

            img_label_path = img_label_path.replace('\\', '/')
            # 数据：图像名称、标注名称、结果名称、检出结果、置信度、标注图、预测图
            label_cls = '正常'
            pred_cls = self.code_dict[str(pred_info[4])]
            conf_cls = pred_info[-1]
            # # debug 临时修改，只过滤STM
            # if not "STM" in pred_cls:
            #     return
            if (label_cls != 'Line0' and label_cls != 'Line1') or (pred_cls != 'Line0' and pred_cls != 'Line1'):
                self.overkill_row += 1
                data = [image_name, label_cls, pred_cls, '过检', conf_cls, img_label_path, pred_image_path]
                self.add_row2write( self.overkill_worksheet, self.overkill_row, data, self.overkill_cell_format)

        elif result_status == 'escape':
            # 绘制
            label_box = label_info[0:4]
            img_label_path = self.draw_circle_label(src, label_box, image_name)
            img_label_path = img_label_path.replace('\\', '/')
            # 数据：图像名称、标注名称、结果名称、检出结果、置信度、标注图、预测图
            label_cls = self.code_dict[str(label_info[-1])]
            pred_cls = '正常'
            conf_cls = 0
            if (label_cls != 'Line0' and label_cls != 'Line1') or (pred_cls != 'Line0' and pred_cls != 'Line1'):
                self.escape_row += 1
                data = [image_name, label_cls, pred_cls, '漏检', conf_cls, img_label_path, pred_image_path]
                self.add_row2write( self.escape_worksheet, self.escape_row, data, self.escape_cell_format)

        elif result_status == 'misclass':
            # 绘制
            label_box = label_info[0:4]
            img_label_path = self.draw_circle_label(src, label_box, image_name)
            img_label_path = img_label_path.replace('\\', '/')
            # 数据：图像名称、标注名称、结果名称、检出结果、置信度、标注图、预测图
            label_cls = self.code_dict[str(label_info[-1])]
            pred_cls = self.code_dict[str(pred_info[4])]
            conf_cls = pred_info[-1]
            
            if (label_cls != 'Line0' and label_cls != 'Line1') or (pred_cls != 'Line0' and pred_cls != 'Line1'):
                self.misclass_row += 1
                data = [image_name, label_cls, pred_cls, '误分类', conf_cls, img_label_path, pred_image_path]
                self.add_row2write( self.misclass_worksheet, self.misclass_row, data, self.misclass_cell_format)

        else:
            # 绘制
            label_box = label_info[0:4]
            img_label_path = self.draw_circle_label(src, label_box, image_name)
            img_label_path = img_label_path.replace('\\', '/')
            # 数据：图像名称、标注名称、结果名称、检出结果、置信度、标注图、预测图
            label_cls = self.code_dict[str(label_info[-1])]
            pred_cls = self.code_dict[str(pred_info[4])]
            conf_cls = pred_info[-1]
            
            if (label_cls != 'Line0' and label_cls != 'Line1') or (pred_cls != 'Line0' and pred_cls != 'Line1'):
                self.ok_row += 1
                data = [image_name, label_cls, pred_cls, '正确检出', conf_cls, img_label_path, pred_image_path]
                self.add_row2write( self.ok_worksheet, self.ok_row, data, self.ok_cell_format)

    def draw_circle_label(self, img, box, image_name):
        cen_x = int((box[0] + box[2]) / 2.0)
        cen_y = int((box[1] + box[3]) / 2.0)
        center = (cen_x, cen_y)

        width = box[2] - box[0]
        height = box[3] - box[1]
        radius = int(max(width, height) / 2 + 10)

        dst = cv2.circle(img, center, radius, (0,0,255), 1, 2)   
        image_name = f"{image_name.split('.')[0]}_center_{cen_x}_{cen_y}_redius_{radius}{img_ext}"
        save_path = join(self.temp_path, image_name)
        cv2.imencode(img_ext, dst)[1].tofile(save_path)
        time.sleep(0.01)
        return save_path

    def load_json_file(self, json_path, source_path):
        print("开始加载json信息")
        if source_path != '':
            if source_path[-4:] == '.txt':
                filenames = []
                with open(source_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip() == '':
                            continue
                        name = line.strip().split('/')[-1].split('.')[0]
                        filenames.append(f"{json_path}/{name}.json")
        else:
            filenames = glob.glob(json_path + '/**/*.json', recursive=True)
        # 遍历json
        for file in tqdm(filenames):
            self.split_json_info(file)
        print("完成加载json信息")


    def split_json_info(self, filename):
        result_status = ''
        pred_info = []
        label_info = []

        with open(filename, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        overkill_list = json_data["Annotations"]["overkill"]
        escape_list = json_data["Annotations"]["escape"]
        misclass_list = json_data["Annotations"]["misclass"]
        ok_list = json_data["Annotations"]["ok"]
        image_name = json_data["Images:"]["name"]
        # 依次解析
        for i in range(len(overkill_list)):
            result_status = 'overkill'
            pred_info = overkill_list[i]["pred_data"]
            label_info = overkill_list[i]["label_data"]
            self.write_data2xlsx(image_name, result_status, pred_info, label_info)

        for i in range(len(escape_list)):
            result_status = 'escape'
            pred_info = escape_list[i]["pred_data"]
            label_info = escape_list[i]["label_data"]
            self.write_data2xlsx(image_name, result_status, pred_info, label_info)
        
        for i in range(len(misclass_list)):
            result_status = 'misclass'
            pred_info = misclass_list[i]["pred_data"]
            label_info = misclass_list[i]["label_data"]
            self.write_data2xlsx(image_name, result_status, pred_info, label_info)

        for i in range(len(ok_list)):
            result_status = 'ok'
            pred_info = ok_list[i]["pred_data"]
            label_info = ok_list[i]["label_data"]
            self.write_data2xlsx(image_name, result_status, pred_info, label_info)


    def close_xlsx(self):        
        # 关闭工作簿
        self.ok_workbook.close()        
        self.overkill_workbook.close()        
        self.escape_workbook.close()
        self.misclass_workbook.close()

    def get_label_box(self, label_file, H, W):
        boxes = []
        colors = []

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                cen_x = float(line.strip().split(' ')[1]) * W
                cen_y = float(line.strip().split(' ')[2]) * H
                width = float(line.strip().split(' ')[3]) * W
                height = float(line.strip().split(' ')[4]) * H

                lt_x = int(cen_x - width / 2.0)
                lt_y = int(cen_y - height / 2.0)
                rb_x = int(cen_x + width / 2.0)
                rb_y = int(cen_y + height / 2.0)

                box = (lt_x, lt_y, rb_x, rb_y)
                boxes.append(box)

                cls = int(line.strip().split(' ')[0])
                color = self.color_dict[str(cls)]
                colors.append(color)

        return boxes, colors
    
    def draw_label2img(self, source_path):
        print('开始绘制标注框')
        # 遍历图像，绘制
        if source_path[-4:] == '.txt':
            filenames = []
            with open(source_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip() == '':
                        continue
                    filenames.append(line.strip())
        else:     
            filenames = glob.glob(source_path + '/**/*'+img_ext, recursive=True)
        for image_file in tqdm(filenames):    
            image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
            label_file = image_file.replace('/images/', '/labels/').replace('.bmp', '.txt')
            # 读取标注文件获取标注框
            boxes, colors = self.get_label_box(label_file, image.shape[0], image.shape[1])
            for i in range(len(boxes)):
                cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), colors[i], 1)
            
            temp_file_name = image_file.split('/')[-1].replace(".bmp", img_ext)
            save_path = join(self.label_images_path, temp_file_name)
            cv2.imencode(img_ext, image)[1].tofile(save_path)
        print('绘制标注框完成')

    def get_pred_box(self, xml_path):
        boxes = []
        colors = []
        mytree = ET.parse(xml_path)
        myroot = mytree.getroot()

        # 获取标注object信息
        for obj in myroot.iter('object'):            
            # box
            box = obj.find('bndbox')
            x_min = int(box.find('xmin').text)
            y_min = int(box.find('ymin').text)
            x_max = int(box.find('xmax').text)
            y_max = int(box.find('ymax').text)
            
            box = (x_min, y_min, x_max, y_max)
            boxes.append(box)

            # label名称：需要区分等级的Label要增加等级
            difficult = obj.find('difficult').text
            name = obj.find('name').text
            code_name = name+str(difficult)
            color = self.color_dict[str(self.code_list.index(code_name))]
            colors.append(color)
        return boxes, colors
    

    def draw_pred2img(self, source_path, json_path):
        print('开始绘制结果框')
         # 遍历图像，绘制
        if source_path[-4:] == '.txt':
            filenames = []
            with open(source_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip() == '':
                        continue
                    filenames.append(line.strip())
        else:     
            filenames = glob.glob(source_path + '/**/*'+img_ext, recursive=True)

        for image_path in tqdm(filenames):
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            name = image_path.split('/')[-1].replace('.bmp', '.xml')
            xml_path = f"{json_path}/{name}"

            boxes, colors = self.get_pred_box(xml_path)
            for i in range(len(boxes)):
                cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), colors[i], 1)
            temp_file_name = image_path.split('/')[-1].replace(".bmp", img_ext)
            save_path = join(self.pred_images_path, temp_file_name)
            cv2.imencode(img_ext, image)[1].tofile(save_path)
        print('绘制结果框完成')

if __name__ == "__main__":
    
    source_list = ['train_0.txt', 'train_1.txt', 'train_2.txt']
    # source_list = ['test_STM.txt']
    for i in range(len(source_list)):
        source_path = r'D:\0_work\dataset_ijp_augment\STM' + '\\' +  source_list[i]
        # 生成的结果json路径
        json_path = r'D:\0_work\report\result\train_0515\result'
        # 存储报告的路径
        report_path = r'D:\0_work\report\STM\train_' + str(i)
        # code_list的路径
        code_path = r'D:\0_work\dataset_ijp_augment\labels.txt'

        xlsx_class = gen_excle_class(report_path, code_path)
        xlsx_class.draw_label2img(source_path)
        xlsx_class.draw_pred2img(source_path, json_path)
        # 先绘制标注
        xlsx_class.load_json_file(json_path, source_path)        
        # 打印个数
        print('正确检出数：', xlsx_class.ok_row)
        print('过检数：', xlsx_class.overkill_row)
        print('漏检数：', xlsx_class.escape_row)
        print('误分类数：', xlsx_class.misclass_row)
        xlsx_class.close_xlsx()

