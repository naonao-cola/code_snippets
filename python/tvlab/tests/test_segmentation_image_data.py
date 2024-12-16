import unittest
import json
import os.path as osp
from tvlab import *
from test_detection_image_data import TestImageBBoxLabelList

with open(osp.normpath('data/coco_instance/instances_val2017_10.json'), "r") as f:
    data = json.load(f)


class TestImagePolygonLabelList(TestImageBBoxLabelList):
    def __init__(self, *args, **kwargs):
        super(TestImagePolygonLabelList, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('data/coco_instance/')
        self.json_path = osp.join(self.data_path, "segmentation_json")
        self.All_NUMBER = 10
        self.ALL_LABELS = ["person", "dog"]
        self.PRE_IMAGE_SHAPE = (427, 640, 3)
        self.image_list = [osp.normpath(x) for x in data["img_path_list"]]
        self.one_image_path = osp.normpath('data/coco_instance/000000329219.jpg')
        self.label_list = data["label_list_ploy"]
        self.none_label_list = [None] * 10
        self.sorted_image_tuple = ('data/coco_instance/000000067213.jpg',
                                   'data/coco_instance/000000193162.jpg',
                                   'data/coco_instance/000000193674.jpg',
                                   'data/coco_instance/000000236166.jpg',
                                   'data/coco_instance/000000329219.jpg',
                                   'data/coco_instance/000000369541.jpg',
                                   'data/coco_instance/000000404484.jpg',
                                   'data/coco_instance/000000419974.jpg',
                                   'data/coco_instance/000000462728.jpg',
                                   'data/coco_instance/000000554002.jpg')
        self.sorted_image_tuple = tuple([osp.normpath(x) for x in self.sorted_image_tuple])
        self.filtered_image_tuple = (osp.normpath('data/coco_instance/000000554002.jpg'),)
        self.split_image_tuple_train = ('data/coco_instance/000000329219.jpg',
                                        'data/coco_instance/000000404484.jpg',
                                        'data/coco_instance/000000369541.jpg',
                                        'data/coco_instance/000000419974.jpg',
                                        'data/coco_instance/000000236166.jpg',)
        self.split_image_tuple_train = tuple([osp.normpath(x) for x in self.split_image_tuple_train])
        self.split_image_tuple_val = ('data/coco_instance/000000462728.jpg',
                                      'data/coco_instance/000000193162.jpg',
                                      'data/coco_instance/000000193674.jpg',
                                      'data/coco_instance/000000067213.jpg',
                                      'data/coco_instance/000000554002.jpg')
        self.split_image_tuple_val = tuple([osp.normpath(x) for x in self.split_image_tuple_val])
        self._cls_ = ImagePolygonLabelList

    def test_label_from_func(self):
        ipll = self._cls_.from_folder(self.data_path)
        ipll.label_from_func(lambda x: {'labels': ['A'], 'bboxes': [[10, 20, 100, 200, 20, 40, 50, 80]]})
        self.assertJsonEqual(ipll.y,
                             [{'labels': ['A'], 'bboxes': [[10, 20, 100, 200, 20, 40, 50, 80]]}] * self.All_NUMBER)

    @unittest.skip("skip test_from_pascal_voc")
    def test_from_pascal_voc(self):
        pass

    def test_from_labelme(self):
        ipll = self._cls_.from_labelme(self.data_path, json_dir=self.json_path)
        self.assertCountEqual(ipll.x, self.image_list)


if __name__ == '__main__':
    unittest.main()
