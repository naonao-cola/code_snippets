from os import path as osp
from tvlab import *
from tvlab.detection import TvdlDetectionTrain
from tvlab.detection import TvdlDetectionInference
import numpy as np
import cv2

__all__ = [
    'StripEtchDetectionTrain',
    'StripDetectionInference']


import imgaug.augmenters as iaa
RESIZE_W, RESIZE_H = 512, 512
iaa_aug_seg = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.1),
    iaa.GaussianBlur(0.1),
    iaa.SaltAndPepper(p=(0.0, 0.001)),
])

iaa_resize = iaa.Resize({'width': RESIZE_W, 'height': RESIZE_H})
g_train_tfms = [iaa_resize, iaa_aug_seg]
g_valid_tfms = [iaa_resize]


class StripEtchDetectionTrain(TvdlDetectionTrain):

    def __init__(self,tmp_dir,need_stop=None):
        super(StripEtchDetectionTrain,self).__init__(tmp_dir)


    def run(self,data_path,training_info,callback =None):
        self.training_info = training_info
        self.data_path = data_path
        self.train_schedule = training_info["train_schedule"]
        self.labelSet = training_info["labelSet"]
        self.cbs =[]
        if callback is not None:
            self.cbs.append(callback)

        self._init_fmt()
        self._init_data()
        super(StripEtchDetectionTrain,self).train(ill=self.ibll,
                                                  train_schedule = self.train_schedule,
                                                  train_tfms=self.train_tfms,
                                                  valid_tfms=self.valid_tfms,
                                                  cbs=self.cbs)


    def _init_fmt(self):
        self.train_tfms = g_train_tfms
        self.valid_tfms = g_valid_tfms

    def _init_data(self):
         #构建数据，拆分为训练集，测试集
        self.image_list =[]
        self.label_list = []
        for item in self.labelSet:
            self.image_list.append(item["imagePath"])
            bboxes = []
            labels = []
            for box in item["boxs"]:
                #字符串转float
                str_list = [float(item) for item in box.split(",")]

                bboxes += [polygon_to_bbox(str_list)]
            for label in item["labels"]:
                labels.append(label)
            y = BBoxLabel({'labels': labels, 'bboxes': bboxes})
            self.label_list.append(y)
        self.ibll = ImageBBoxLabelList(self.image_list, self.label_list)
        self.ibll_t, self.ibll_v = self.ibll.split(0.2,show=False)
        self.classes = self.ibll.labelset()


    def _init_train_schedule(self):
        """
        """
        pass

    def evaluate(self,result_path):
        super(StripEtchDetectionTrain,self).evaluate(result_path=result_path, iou_threshold=0.5,bboxes_only=False, callback=None, box_tfm=None)


    def package_model(self,model_path, import_cmd = "",callback=None):
        super(StripEtchDetectionTrain,self).package_model(model_path = model_path,
                                                        #model_fmt='ckpt',
                                                        import_cmd = import_cmd
                                                        )
        pass


def box_tfm(box, ori_shape):
    h, w = ori_shape[:2]
    h_scale, w_scale = h*1.0/RESIZE_H, w*1.0/RESIZE_W
    l,t,r,b = box
    return [l*w_scale, t*h_scale, r * w_scale, b*h_scale]


class StripDetectionInference(TvdlDetectionInference):
    def run(self,image_list,extra_info=None,top_rank =4,filter_conf =0.3):
        self.image_list = image_list
        self.ibll = ImageBBoxLabelList(self.image_list)
        self.ibll.split(valid_pct=1.0,show=False)
        self.train_tfms= g_train_tfms
        self.valid_tfms = g_valid_tfms
        if self.model is None:
            self.load_model()
        y_pred = self.predict(self.ibll, tfms=self.valid_tfms , box_tfm=box_tfm)

        ret = {}
        sub_code={}
        for x,y in zip(image_list,y_pred):
            sub_code[str(x)]={}
            #全部记为数组，之后再字符串操作

            org_code = []
            for label_item in y["labels"]:
                org_code.append(label_item)
            org_conf =[]
            org_boxes =[]
            org_boxesSize =[]
            for box_item in y["bboxes"]:
                box_width = int(box_item[2] - box_item[0])
                box_hight = int(box_item[3] - box_item[1])
                box_area = int(box_width * box_hight)

                box_str = str(int(box_item[0])) + "," + str(int(box_item[1])) + "," + str(int(box_item[2])) + "," + str(int(box_item[3]))
                boxsize_str = str(box_width) +"," +str(box_hight) +","+str(box_area)

                org_boxesSize.append(boxsize_str)
                org_boxes.append(box_str)
                org_conf.append(box_item[4])

            # org_conf 排序,得到排序后的索引
            sorted_id = sorted(range(len(org_conf)), key=lambda k: org_conf[k], reverse=True)

            code =[]
            conf=[]
            boxes=[]
            boxesSize =[]

            if len(sorted_id) <=top_rank:
                for i in range(len(sorted_id)):
                    if i >= len(sorted_id):
                        break
                    if org_conf[sorted_id[i]] < filter_conf:
                        top_rank = top_rank +1
                        continue
                    code.append(org_code[sorted_id[i]])
                    conf.append(str(org_conf[sorted_id[i]]))
                    boxes.append(org_boxes[sorted_id[i]])
                    boxesSize.append(org_boxesSize[sorted_id[i]])
            else:
                for i in range(top_rank):
                    if i >= len(sorted_id):
                        break
                    if org_conf[sorted_id[i]] < filter_conf:
                        top_rank = top_rank +1
                        continue
                    code.append(org_code[sorted_id[i]])
                    conf.append(str(org_conf[sorted_id[i]]))
                    boxes.append(org_boxes[sorted_id[i]])
                    boxesSize.append(org_boxesSize[sorted_id[i]])

            code = ";".join(code)
            conf = ";".join(conf)
            boxesSize = ";".join(boxesSize)

            image = cv2.imread(x)
            size = image.shape
            image_w = size[1]
            iimage_h = size[0]

            sub_code[x]["imageSize"] =str(image_w) +","+str(iimage_h)
            sub_code[x]["lightness"] = conf
            sub_code[x]["boxesSize"] = boxesSize
            sub_code[x]["code"] = code
            sub_code[x]["boxes"] = boxes
            sub_code[x]["conf"] = conf
            sub_code[x]["status"]="OK"

        ret["sub_code"] = sub_code
        ret["main_code"] ={}
        return ret