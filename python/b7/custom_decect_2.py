### custom_decect_2.py

'''
1、加载参数
2、读取样本：小图、大图（切图）
3、执行推理
4、结果存储（json）：小图、大图（合并）
5、绘制、存图
6、统计过杀漏检等
'''

import argparse
from math import ceil
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import yaml
from tqdm import tqdm
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom
import json
import copy


img_ext = ".jpg"
ignore_cls_index = [1,9]
# 同类型code不同级别时，进行合并,key为主code，
BBOX_MERGE_CLS_INFO={3: [2]}
SAME_CLS_BBOX_MERGE_CLS_INFO = [2]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp22/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/data/tvt/tdg/Dev/yolov5-6.1/data/my_ijp_yolo_0607_stm_only.yaml', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/my_ijp_yolo.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_path', default= '/data/tvt/dataset_ijp_augment/train_exp22/result', help='')

    # 标签文件    
    parser.add_argument('--save_json', default= True, help='')
    parser.add_argument('--label_source', default= '', help='')

    # 图像增强 // 测试本地图不用二次增强
    parser.add_argument('--enable_augImg', default = False, help='')
    parser.add_argument('--alpha', default = -1, help='')
    parser.add_argument('--beta', default = 0 , help='')
   
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print("data", opt.data)
    return opt

def is_overlap(box1, box2):
    return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])

def merge_bboxes(boxes):
    return [min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes)]


# 存储标注信息
class label_info:
    def __init__(self, name, H, W):     
        self.name = name
        self.H = H
        self.W = W

    def set_info(self, box, cls):
        self.box = box
        self.cls = cls

    def split_info(self, label_file):
        boxes = []
        clses = []

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                cen_x = float(line.strip().split(' ')[1]) * self.W
                cen_y = float(line.strip().split(' ')[2]) * self.H
                width = float(line.strip().split(' ')[3]) * self.W
                height = float(line.strip().split(' ')[4]) * self.H

                lt_x = int(cen_x - width / 2.0)
                lt_y = int(cen_y - height / 2.0)
                rb_x = int(cen_x + width / 2.0)
                rb_y = int(cen_y + height / 2.0)

                box = [lt_x, lt_y, rb_x, rb_y]
                boxes.append(box)
                clses.append(int(line.strip().split(' ')[0]))
        self.set_info(boxes, clses)
        

# 存储推理信息
class pred_info:
    def __init__(self, box, cls, conf, name):
        # result: ok\escape\overkill\misclass
        self.box = box
        self.cls = cls
        self.conf = conf
        self.name = name

# 存储所有推理结果信息
class dataset_result:
    def __init__(self):
        # 初始化字典变量
        self.result_dict = {"Images":[], "Annotations":[]}
        self.Images_dict = {"name":"", "H":-1, "W":-1}

        self.overkill_list = []
        self.escape_list = []
        self.misclass_list = []
        self.ok_list = []

        self.Annotations_dict = {"overkill":[], "escape":[], "misclass":[], "ok":[]}
        self.overkill_dict = {"pred_data":[], "label_data":[]}
        self.escape_dict = {"pred_data":[], "label_data":[]}
        self.misclass_dict = {"pred_data":[], "label_data":[]}
        self.ok_dict = {"pred_data":[], "label_data":[]}
    
    def dump_json(self, path, filename):
        self.Annotations_dict = {"overkill":self.overkill_list, "escape":self.escape_list, "misclass":self.misclass_list, "ok":self.ok_list}
        self.result_dict = {"Images:":self.Images_dict, "Annotations":self.Annotations_dict}
        
        json_path = f"{path}/{filename.split('.')[0]}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.result_dict, f, indent=4)


class custom_yolov5_infer():

    # 加载参数
    def __init__(self, opt):
        self.opt = opt

        # 参数 加载模型       
        self.device = select_device(self.opt.device)
        dnn = False
        data = self.opt.data
        weights = self.opt.weights
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = self.opt.imgsz
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # Half
        self.half = False
        self.half &= (pt) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=self.half)  # warmup
        self.names = names

    # 读取样本
    def load_sample(self):
        data_path = self.opt.source
        print(data_path)
        file_content = []
        if data_path[-4:] == 'yaml':
            with open(data_path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.load(file, Loader=yaml.FullLoader)
            yaml_path = yaml_data['path']
            yaml_test = yaml_data['test']
            test_path = f"{yaml_path}/{yaml_test}"
            with open(test_path, 'r', encoding='utf-8') as file:                
                for line in file:
                    file_content.append(line.strip())
        else:
            for temp_file in os.listdir(data_path):
                file_content.append(f"{data_path}/{temp_file}")
        return file_content

    # 图像增强
    def clahe(self, src):
        clahe_exec = cv2.createCLAHE(clipLimit=8.0)
        if len(src.shape) == 2:  # 单通道灰度图像
            enhanced_img = clahe_exec.apply(src)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
        elif len(src.shape) == 3:  # 三通道彩色图像
            # 分离通道
            channels = cv2.split(src)            
            # 对每个通道分别应用 CLAHE
            enhanced_0 = clahe_exec.apply(channels[0])
            enhanced_1 = clahe_exec.apply(channels[1])
            enhanced_2 = clahe_exec.apply(channels[2])            
            # 合并通道
            enhanced_img = cv2.merge((enhanced_0, enhanced_1, enhanced_2))   
        return enhanced_img

    def adjust_gamma(self, src, gamma = 1.0):
        if gamma == 0:
            gamma = 0.00001        
        if gamma < 0:
            gamma = 1 + (gamma - 1)
        inv_gamma = gamma
        dst = np.zeros(src.shape, dtype=np.uint8)
        # lut_data = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        lut_data = np.array([((i / 255.0) ** inv_gamma) * 255 if i != 0 else 0 for i in range(256)], dtype=np.uint8)
        cv2.LUT(src, lut_data, dst)
        return dst

        # 图像增强
    def augment_image_old(self, img):   
        clipLimit = 10
        tileGridSize = (8, 8)
        # clipLimit = self.opt.clipLimit
        # tileGridSize = self.opt.tileGridSize

        if len(img.shape) == 2:  # 单通道灰度图像
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            enhanced_img = clahe.apply(img)
            enhanced_img = cv2.cvtCOLOR(enhanced_img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:  # 三通道彩色图像
            # 分离通道
            b, g, r = cv2.split(img)            
            # 对每个通道分别应用 CLAHE
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            enhanced_b = clahe.apply(b)
            enhanced_g = clahe.apply(g)
            enhanced_r = clahe.apply(r)            
            # 合并通道
            enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
        return enhanced_img
    
    
    def augment_image(self, src):        
        alpha = self.opt.alpha
        beta = self.opt.beta

        MAX_CONTRAST = 300
        MAX_BRIGHTNESS=255
        alpha = (alpha + MAX_CONTRAST) / MAX_CONTRAST
        beta = (beta + MAX_BRIGHTNESS) / MAX_BRIGHTNESS

        src2 = np.zeros(src.shape, dtype=np.uint8)
        gamma_img = np.zeros(src.shape, dtype=np.uint8)
        result = np.zeros(src.shape, dtype=np.uint8)

        if alpha != 1.0:
            clahe_img = self.clahe(src) 
        else:
            cv2.addWeighted(src, beta, clahe_img, 1, 1, clahe_img, -1)
        light_img = np.zeros(src.shape, dtype=np.uint8)
        cv2.addWeighted(clahe_img, 1, src2, beta, 1, light_img, -1)
        gamma_img = self.adjust_gamma(light_img, alpha)
        cv2.addWeighted(gamma_img, 1, src2, 1, 1, result, -1)
        return gamma_img

    def exec_detector(self):
        # 读取样本
        file_content = self.load_sample()     
        enable_augImg = self.opt.enable_augImg
        # 遍历图像
        for file in tqdm(file_content):
            if file == '':
                continue
            image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)

            # 图像预处理
            t1 = time.time()

            if enable_augImg:
                image = self.augment_image(image)
            t2 = time.time()
            xyxy_results = []
            conf_results = []
            cls_results = []

            if image.shape[0] == self.opt.imgsz[0]:
                # Convert
                img = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                im = torch.from_numpy(img).to(self.device)
                im = im.half() if self.half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t3 = time.time()

                # 执行推理
                pred = self.exec_infer(im)
                t4 = time.time()
                for i, det in enumerate(pred):
                    for *xyxy, conf, cls in reversed(det):
                        temp_xyxy = []
                        for _xyxy in xyxy:
                            temp_xyxy.append(int(_xyxy.cpu().detach()))
                        xyxy_results.append(temp_xyxy)
                        conf_results.append(float(conf.cpu().detach()))
                        cls_results.append(int(cls.cpu().detach()))
                t5 = time.time()
                loginfo = f"augment image: {(t2 - t1)*1000:0.3f}ms, preprocess: {(t3 - t2)*1000:0.3f}ms, inference: {(t4 - t3)*1000:0.3f}ms, postprocess: {(t5 - t4)*1000:0.3f}ms"
                print(loginfo)

            else:   # 切小图
                t3 = time.time()
                model_size = self.opt.imgsz[0]
                overlap = 0.25
                overlap_size = int(overlap / 2.0 * model_size)

                (h, w, c) = image.shape
                h_index, w_index = ceil(h/model_size), ceil(w/model_size)
                start_x = 0
                start_y = 0
                xyxy_rst = []
                conf_rst = []
                cls_rst = []
                
                for i in range(h_index):
                    for j in range(w_index):
                        start_y = max(int(i * model_size - overlap_size*i), 0)
                        start_x = max(int(j * model_size - overlap_size*j), 0)
                        end_x = start_x + model_size
                        end_y = start_y + model_size
                        # 超限位置
                        if end_x > w:
                            over_size = end_x - w
                            end_x = w
                            start_x = start_x - over_size
                        if end_y > h:
                            over_size = end_y -h
                            end_y = h
                            start_y = start_y - over_size

                        # 截图
                        img_tailor = image[int(start_y):int(end_y),int(start_x):int(end_x)]
                        # 保存验证
                        # img_tailor_img = Image.fromarray(img_tailor)
                        # img_tailor_img.save("/data/tvt/tdg/Dev/OriginData/Online/testcrop/img_"+str(i)+str(j)+".jpg");
                        

                        img = img_tailor.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                        img = np.ascontiguousarray(img)
                        im = torch.from_numpy(img).to(self.device)
                        im = im.half() if self.half else im.float()  # uint8 to fp16/32
                        im /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(im.shape) == 3:
                            im = im[None]  # expand for batch dim                        
                        # 执行推理
                        pred = self.exec_infer(im)
                        for _, det in enumerate(pred):
                            for *xyxy, conf, cls in reversed(det):
                                cls_type = int(cls.cpu().detach())
                                if cls_type in ignore_cls_index:
                                    continue
                                temp_xyxy = []                                
                                temp_xyxy.append(int(xyxy[0].cpu().detach() + start_x))
                                temp_xyxy.append(int(xyxy[1].cpu().detach() + start_y))
                                temp_xyxy.append(int(xyxy[2].cpu().detach() + start_x))
                                temp_xyxy.append(int(xyxy[3].cpu().detach() + start_y))

                                xyxy_rst.append(temp_xyxy)
                                conf_rst.append(float(conf.cpu().detach()))
                                cls_rst.append(int(cls.cpu().detach()))
              
                # 缺陷结果合并
                t4 = time.time()
                xyxy_results, conf_results, cls_results = self.merge_result(xyxy_rst, conf_rst, cls_rst)

                t5 = time.time()
                loginfo = f"augment image: {(t2 - t1)*1000:0.3f}ms, crop images and inference: {(t4 - t3)*1000:0.3f}ms, merge result: {(t5 - t4)*1000:0.3f}ms"
                print(loginfo)
            # 1、绘制存储
            concat_save_path = f"{self.opt.save_path}"
            # if len(cls_results) == 0:
            #     concat_save_path = f"{self.opt.save_path}/OK"
            # else:
            #     concat_save_path = f"{self.opt.save_path}/NG"
            xyxy_results, conf_results, cls_results = self.bbox_merge_cls((xyxy_results, conf_results, cls_results))
            xyxy_results, conf_results, cls_results = self.bbox_merge_same_cls((xyxy_results, conf_results, cls_results))
            
            xml_folder = file
            xml_filename = file.split('/')[-1]
            
            xml_filename = xml_filename.replace('.bmp', '.jpg')
            
            img_name = file.split('/')[-1].split('.')[0]
            img_name = img_name + img_ext
            self.draw_result(image, xyxy_results, conf_results, cls_results, img_name, concat_save_path)

            # 2、存储标注信息
            self.dump_xml(xml_folder, xml_filename, image.shape, xyxy_results, conf_results, cls_results, concat_save_path)

            # 3、存储json信息    
            if self.opt.save_json:
                json_object = dataset_result()
                json_object.Images_dict["name"] = xml_filename
                json_object.Images_dict["H"] = image.shape[0]
                json_object.Images_dict["W"] = image.shape[1]

                 # 执行结果json存储
                label_source = self.opt.label_source
                if label_source != '':                    
                    label_path = f"{label_source}/{file.split('/')[-1].replace('.bmp', '.txt')}"
                else:
                    label_path = file.replace('/images/', '/labels/')
                    label_path = label_path.replace('.bmp', '.txt')

                # 解析label
                label_inf = label_info(file.split('/')[-1], image.shape[0], image.shape[1])
                label_inf.split_info(label_path)
                pred_inf = pred_info(xyxy_results, cls_results, conf_results, file.split('/')[-1])

                # 计算检测结果
                self.calc_result(label_inf, pred_inf, json_object)

                json_object.dump_json(concat_save_path, xml_filename)
                
    def bbox_merge_same_cls(self, merge_preds):
        xyxy_results, conf_results, cls_results = merge_preds
        to_delete = set()
        merged_boxes = []
        merged_confidences = []
        merged_codes = []
        
        for same_cls in SAME_CLS_BBOX_MERGE_CLS_INFO:
            same_code_indices = [i for i in range(len(cls_results)) if cls_results[i] == same_cls]
            for cur_same_idx, j in enumerate(same_code_indices):
                # 遍历找到的数据，进行box iou处理
                # 需要
                inter_boxes =[xyxy_results[j]]
                inter_confs = [conf_results[j]]
                # 遍历对比
                for k in same_code_indices[cur_same_idx:]:
                    if is_overlap(xyxy_results[k], xyxy_results[j]):
                        inter_boxes.append(xyxy_results[k])
                        inter_confs.append(conf_results[k])
                        to_delete.add(k)
                        
                if len(inter_boxes) > 1:
                    merged_box = merge_bboxes(inter_boxes)
                    merged_boxes.append(merged_box)
                    merged_confidences.append(conf_results[j])
                    merged_codes.append(same_cls)
                    to_delete.add(j)
            
            # 准备结果列表
            new_a = []
            new_b = []
            new_c = []

            # 添加未删除的原始框
            for i in range(len(xyxy_results)):
                if i not in to_delete:
                    new_a.append(xyxy_results[i])
                    new_b.append(conf_results[i])
                    new_c.append(cls_results[i])

            # 添加合并后的框
            new_a.extend(merged_boxes)
            new_b.extend(merged_confidences)
            new_c.extend(merged_codes)
        
        return new_a, new_b, new_c

    def bbox_merge_cls(self, merge_preds):
        '''
        同类型code合并框，以merge_info为配置进行合并
        '''
        xyxy_results, conf_results, cls_results = merge_preds
        to_delete = set()
        merged_boxes = []
        merged_confidences = []
        merged_codes = []
        
        for key,vals  in BBOX_MERGE_CLS_INFO.items():
            # 找到 code 为 2 和 3 的框
            code_sub_indices = [i for i in range(len(cls_results)) if cls_results[i] in vals]
            code_main_indices = [i for i in range(len(cls_results)) if cls_results[i] == key]

            for j in code_main_indices:
                
                inter_boxes =[xyxy_results[j]]
                inter_confs = [conf_results[j]]
                
                for i in code_sub_indices:
                    if is_overlap(xyxy_results[i], xyxy_results[j]):
                        inter_boxes.append(xyxy_results[i])
                        inter_confs.append(conf_results[i])
                        to_delete.add(i)
                        
                if len(inter_boxes) > 1:
                    merged_box = merge_bboxes(inter_boxes)
                    merged_boxes.append(merged_box)
                    merged_confidences.append(conf_results[j])
                    merged_codes.append(key)
                    to_delete.add(j)
            
            # 准备结果列表
            new_a = []
            new_b = []
            new_c = []

            # 添加未删除的原始框
            for i in range(len(xyxy_results)):
                if i not in to_delete:
                    new_a.append(xyxy_results[i])
                    new_b.append(conf_results[i])
                    new_c.append(cls_results[i])

            # 添加合并后的框
            new_a.extend(merged_boxes)
            new_b.extend(merged_confidences)
            new_c.extend(merged_codes)
        
        return new_a, new_b, new_c
       
    
    # 执行推理
    def exec_infer(self, im):
        visualize = False
        augment = False
        pred = self.model(im, augment=augment, visualize=visualize)        
        # NMS
        conf_thres = self.opt.conf_thres
        iou_thres = self.opt.iou_thres
        max_det = self.opt.max_det
        classes = None
        agnostic_nms = False
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        return pred

    # 存储标注文件
    def dump_xml(self, xml_folder, xml_filename, shape, xyxy_results, conf_results, cls_results, concat_save_path):        
        annotation = ET.Element("annotation")
        # 添加子元素        
        folder = ET.SubElement(annotation, "folder")
        folder.text = xml_folder

        filename = ET.SubElement(annotation, "filename")
        filename.text = xml_filename

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = str(0)

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = 'Unknown'

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(shape[1])
        height = ET.SubElement(size, "height")
        height.text = str(shape[0])
        depth = ET.SubElement(size, "depth")
        depth.text = str(shape[2])

        # 填充标注信息
        for i in range(len(cls_results)):
            label = self.names[cls_results[i]]
            m_name = label[0:-1]
            m_difficult = label[-1]
            m_subConf = round(conf_results[i], 3)
            m_xmin = xyxy_results[i][0]
            m_ymin = xyxy_results[i][1]
            m_xmax = xyxy_results[i][2]
            m_ymax = xyxy_results[i][3]

            m_object = ET.SubElement(annotation, "object")
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

        # 将 XML 结构保存为文件
        save_path = concat_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        xml_file = xml_filename.split('.')[0]
        xml_filepath = f"{save_path}/{xml_file}.xml"
        
        tree = ET.ElementTree(annotation)
        tree.write(xml_filepath, encoding="utf-8", xml_declaration=True)

        # 使用 xml.dom.minidom 格式化 XML 文件
        dom = xml.dom.minidom.parse(xml_filepath)
        with open(xml_filepath, "w", encoding="utf-8") as f:
            f.write(dom.toprettyxml(indent="    "))  # 使用四个空格作为缩进


    # 绘制结果
    def draw_result(self, image, xyxy_rst, conf_rst, cls_rst, img_name, concat_save_path):        
        save_path = concat_save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_save_path = f"{save_path}/{img_name}"

        line_thickness = 1
        annotator = Annotator(image, line_width=line_thickness, example=str(self.names))
        for i in range(len(cls_rst)):
            c = int(cls_rst[i])  # integer class
            label = f'{self.names[c]} {conf_rst[i]:.2f}'
            annotator.box_label(xyxy_rst[i], label, color=colors(c, True))   
            image = annotator.result()
        cv2.imencode(img_ext, image)[1].tofile(image_save_path)

    # 合并结果
    def merge_result(self, xyxy_rst, conf_rst, cls_rst):
        xyxy_results = []
        conf_results = []
        cls_results = []

        # 类别数组
        # cls_list = cls_rst.unique()
        cls_list = np.unique(cls_rst)
        cls_num = len(cls_list)
        xyxy_unique = [[] for j in range(cls_num)]
        conf_unique = [[] for j in range(cls_num)]
        cls_unique = [[] for j in range(cls_num)]        

        # 按类别分组
        for c in range(cls_num):
            for i in range(len(cls_rst)):
                if cls_rst[i] == cls_list[c]:
                    xyxy_unique[c].append(xyxy_rst[i])
                    conf_unique[c].append(conf_rst[i])
                    cls_unique[c].append(cls_rst[i])

        # 按类别合并: 计算iou
        for i in range(cls_num):
            box_result, conf_result, cls_result = self.custom_nms(xyxy_unique[i], conf_unique[i], cls_unique[i], 0.15)
            xyxy_results.append(box_result)
            conf_results.append(conf_result)
            cls_results.append(cls_result)
        
        # 将数据展平
        xyxy_results_ = [element for sublist in xyxy_results for element in sublist]
        conf_results_ = [element for sublist in conf_results for element in sublist]
        cls_results_ = [element for sublist in cls_results for element in sublist]
        return xyxy_results_, conf_results_, cls_results_

    def custom_nms(self, boxes, confs, cls, iou_thresh):
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = np.array(confs)

        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        # keep_boxes 用于存放执行 NMS 后剩余的 boxes
        keep_boxes = []
        index = scores.argsort()[::-1]
        box_result = []
        conf_result = []
        cls_result = []

        while len(index) > 0:
            i = index[0]
            keep_boxes.append(i)
            x1_overlap = np.maximum(x1[i], x1[index[1:]])
            y1_overlap = np.maximum(y1[i], y1[index[1:]])
            x2_overlap = np.minimum(x2[i], x2[index[1:]])
            y2_overlap = np.minimum(y2[i], y2[index[1:]])

            # 计算重叠部分的面积，若没有不重叠部分则面积为 0
            w = np.maximum(0, x2_overlap - x1_overlap + 1)
            h = np.maximum(0, y2_overlap - y1_overlap + 1)
            overlap_area = w * h
            ious = overlap_area / (areas[i] + areas[index[1:]] - overlap_area)

            idx = np.where(ious <= iou_thresh)[0]

            # 合并大框
            big_idx = np.where(ious > iou_thresh)[0]
            union_index = index[big_idx + 1]
            if union_index.size == 1:
                lt_x = np.minimum(boxes[i][0], boxes[:, 0][union_index])[0]
                lt_y = np.minimum(boxes[i][1], boxes[:, 1][union_index])[0]
                rb_x = np.maximum(boxes[i][2], boxes[:, 2][union_index])[0]
                rb_y = np.maximum(boxes[i][3], boxes[:, 3][union_index])[0]
            elif union_index.size > 1:
                lt_x = np.min(np.minimum(boxes[i][0], boxes[:, 0][union_index]))
                lt_y = np.min(np.minimum(boxes[i][1], boxes[:, 1][union_index]))
                rb_x = np.max(np.maximum(boxes[i][2], boxes[:, 2][union_index]))
                rb_y = np.max(np.maximum(boxes[i][3], boxes[:, 3][union_index]))
            else:
                lt_x = boxes[i][0]
                lt_y = boxes[i][1]
                rb_x = boxes[i][2]
                rb_y = boxes[i][3]

            box_result.append([lt_x, lt_y, rb_x, rb_y])
            conf_result.append(scores[i])
            cls_result.append(cls[i])

            index = index[idx + 1]
        return box_result, conf_result, cls_result
   
    def calc_result(self, label_info, pred_info, json_object):
        img_ok = 0
        img_escape = 0
        img_overkill = 0
        img_misclass = 0

        # label
        label_box = label_info.box
        label_cls = label_info.cls
        # pred
        pred_box = pred_info.box
        pred_cls = pred_info.cls
        pred_conf = pred_info.conf

        # 正确检出、误分类、漏检
        for i in range(len(label_box)):
            is_escape = True # 默认漏检

            l_box = label_box[i]
            l_cls = label_cls[i]
            
            label_list = l_box.copy()
            label_list.append(l_cls)

            for j in range(len(pred_box)):
                p_box = pred_box[j]
                p_cls = pred_cls[j]
                p_conf = round(pred_conf[j], 4)

                # 重组信息
                pred_list = p_box.copy()
                pred_list.append(p_cls)
                pred_list.append(p_conf)

                iou = self.calc_box_iou(p_box, l_box)
                if iou > 0.2:
                    is_escape = False
                    if p_cls == l_cls:
                        img_ok += 1
                        json_object.ok_dict["pred_data"] = pred_list.copy()
                        json_object.ok_dict["label_data"] = label_list.copy()
                        json_object.ok_list.append(copy.deepcopy(json_object.ok_dict))
                    else:
                        img_misclass += 1
                        json_object.misclass_dict["pred_data"] = pred_list.copy()
                        json_object.misclass_dict["label_data"] = label_list.copy()
                        json_object.misclass_list.append(copy.deepcopy(json_object.misclass_dict))

            if is_escape:
                img_escape += 1
                json_object.escape_dict["pred_data"] = []
                json_object.escape_dict["label_data"] = label_list.copy()
                json_object.escape_list.append(copy.deepcopy(json_object.escape_dict))

        # 过杀
        for i in range(len(pred_box)):
            is_overkill = True
            p_box = pred_box[i]
            p_cls = pred_cls[i]
            p_conf = round(pred_conf[i], 4)
            # 重组信息
            pred_list = p_box.copy()
            pred_list.append(p_cls)
            pred_list.append(p_conf)

            for j in range(len(label_box)):
                l_box = label_box[j]
                l_cls = label_cls[j]

                iou = self.calc_box_iou(p_box, l_box)
                if iou > 0.2:
                    is_overkill = False
            
            if is_overkill:
                img_overkill += 1
                json_object.overkill_dict["pred_data"] = pred_list.copy()
                json_object.overkill_dict["label_data"] = []
                json_object.overkill_list.append(copy.deepcopy(json_object.overkill_dict))



    # 计算IOU
    def calc_box_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

         # 计算交集部分的坐标
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # 计算交集面积
        intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
        
        # 计算两个边界框的面积
        area_box1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        area_box2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
        
        # 计算并集面积
        union_area = area_box1 + area_box2 - intersection_area
        
        # 计算 IoU
        iou = intersection_area / union_area    
        return iou

    
def main():
    opt = parse_opt()
    my_infer = custom_yolov5_infer(opt)
    my_infer.exec_detector()    

if __name__ == "__main__":
    main()