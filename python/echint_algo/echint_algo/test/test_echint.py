
import sys
sys.path.append("..")

from tvlab import *
import os,glob,json
from core.echint_algo_main import StripEtchDetectionTrain,StripDetectionInference
import xml.etree.ElementTree as ET
from core import echint_algo_main


# 全部数据集
g_labelSet = []
g_label_calss =set()


#json 文件
json_file = glob.glob("/data/proj/echint/ADC算法POC图片/M1_ETCH/AD61-AD65/*/*.json")
jpg_img_file = [x[:-4]+"jpg" for x in json_file]

def label_func(json_file_list,img_file_list):
    for  json_item,img_item in zip(json_file_list,img_file_list):
        with open(json_item,'r')as f:
            json_data = json.load(f)
        bboxes = []
        labels = []
        shapes = json_data['shapes']
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            if label == "Cu_Oxidize":
                label = "Cu_Plate_Oxidize"
            if label =="Cu_Snow":
                label = "Cu_Snowflake"
            if label =="Metal_LineWide":
                label= "Metal_Line_Wide"
            if label == "Metal_loss":
                label ="Metal_Loss"
            if label == "Metal_Residue":
                # print(img_item)
                continue
            labels.append(label)
            g_label_calss.add(label)
            box = polygon_to_bbox(points)
            # box_str = [str(item) for item in box]
            box_str=""
            for item in box:
                item_str = str(item)
                box_str += item_str +","
            sub_str = box_str[0:-1]
            # print(sub_str)
            bboxes.append(sub_str)
        g_labelSet.append({"imageName":os.path.basename(img_item),
                        "imagePath":img_item,
                        "boxs":bboxes,
                        "labels": labels})

    return g_labelSet




def fix_name(org_list:list,dst_list:list):
    org_file_list = {}
    ret_list =[]

    for i ,file_path in enumerate(org_list):
        portion= os.path.split(file_path)
        # 文件名，路径，
        org_file_list[portion[1]]=portion[0]

    for i,file_path in enumerate(dst_list):
        file_name = os.path.basename(file_path)
        if file_name in org_file_list:

            file_path = str(org_file_list[file_name]) +str("/")+str(file_name)
            ret_list.append(file_path)
    # print(len(ret_list))
    return ret_list

#xml 文件以及对应的图片
xml_files = glob.glob("/data/proj/echint/ADC算法POC图片/M1_ETCH/AD60第一次传图_标注/AD60第一次传图_标注xml/AD6_M1_ETCH_Label/*.xml")
xml_jpg_img_file = [x[:-3]+"jpg" for x in xml_files]
xml_img_files=glob.glob("/data/proj/echint/ADC算法POC图片/M1_ETCH/AD60第一次传图/AD60第一次传图/*/*.jpg")
xml_det_img_list = fix_name(xml_img_files,xml_jpg_img_file)
print("xml_1 文件", len(xml_files))
print("xml_1 图片", len(xml_det_img_list))

# xml 2 文件以及对应的图片
xml_file_2 = glob.glob("/data/proj/echint/ADC算法POC图片/AD87-89/M1_ETCH/*/*.xml")
xml_jpg_from_xml_2 =  [x[:-3]+"jpg" for x in xml_file_2]
xml_img_file_2 = glob.glob("/data/proj/echint/ADC算法POC图片/AD87-89/M1_ETCH/*/*.jpg")
xml_det_img_list_2 = fix_name(xml_img_file_2,xml_jpg_from_xml_2)
print("xml_2 文件",len(xml_file_2))
print("xml_2 图片",len(xml_det_img_list_2))


def xml_func(xml_path_list,img_path_list):
    for xml_path, img_path in zip(xml_path_list,img_path_list):
        bboxes = []
        labels = []
        tree = ET.parse(xml_path)
        for i in range (len(tree.findall("object"))):
            points =[]
            label = tree.findall("object")[i].find("name").text
            xmin = tree.findall("object")[i].find("bndbox")[0].text
            ymin = tree.findall("object")[i].find("bndbox")[1].text
            xmax = tree.findall("object")[i].find("bndbox")[2].text
            ymax = tree.findall("object")[i].find("bndbox")[3].text
            points.append(int(xmin))
            points.append(int(ymin))
            points.append(int(xmax))
            points.append(int(ymax))
            if label =="Cu_Snow":
                label = "Cu_Snowflake"
            if label =="Metal_LineWide":
                label= "Metal_Line_Wide"
            if label == "Cu_Oxida":
                label = "Cu_Plate_Oxidize"
            if label == "RDL_short":
                label = "RDL_Short"
            if label == "Metal_loss":
                label = "Metal_Loss"
            if label == "other":
                label ="Other"
            if label == "CU_Peeling":
                label = "Cu_Peeling"
            if label =="Metal_LineThin":
                label = "Metal_Line_Thin"

            labels.append(label)
            g_label_calss.add(label)
            box = polygon_to_bbox(points)
            box_str=""
            for item in box:
                item_str = str(item)
                box_str += item_str +","
            sub_str = box_str[0:-1]
            bboxes.append(sub_str)

        g_labelSet.append({"imageName":os.path.basename(img_path),
                        "imagePath":img_path,
                        "boxs":bboxes,
                        "labels": labels})

    return g_labelSet



g_labelSet = label_func(json_file,jpg_img_file)
print(g_label_calss)
print(len(g_labelSet))

g_labelSet = xml_func(xml_files,xml_det_img_list)
print(g_label_calss)
print(len(g_labelSet))

g_labelSet = xml_func(xml_file_2,xml_det_img_list_2)
print(g_label_calss)
print(len(g_labelSet))

# 训练参数
training_info={}
training_info ["train_schedule"] = {
                "backbone": "resnet50",
                "monitor": "loss",
                'epochs': 40,
        }
training_info["downloadPart"] ="ok"
training_info["labelSet"] =g_labelSet


# 训练接口测试
train_obj = StripEtchDetectionTrain("/data/proj/www/tmp_echint")
train_obj.run(data_path =None, training_info = training_info)
train_obj.package_model(model_path = "/data/proj/www/tmp_echint/model/echint_test.capp",
                        import_cmd= "from echint_algo import StripDetectionInference"
                        )

# print(training_info)

# training_info_str = json.dumps(training_info)
# with open('/data/proj/www/work/strip/test/training_info.json', 'w') as json_file:
#     json_file.write(training_info_str)

#推理接口
inference_obj = StripDetectionInference(model_path = "/data/proj/www/tmp_echint/model/echint_test.capp",
                                        work_dir = "/data/proj/www/tmp_echint")
test_img_list = xml_det_img_list[:5]
pred = inference_obj.run(image_list = test_img_list)

# print(pred)
pred_info_str = json.dumps(pred)
with open('/data/proj/www/work/strip/test/pred_info.json', 'w') as json_file:
    json_file.write(pred_info_str)
