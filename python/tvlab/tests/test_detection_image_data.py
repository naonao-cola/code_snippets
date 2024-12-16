import unittest
import json
import os.path as osp
from tvlab import *
from test_category_image_data import TestImageLabelList

with open(osp.normpath('data/coco_instance/instances_val2017_10.json'), "r") as f:
    data = json.load(f)


class TestImageBBoxLabelList(TestImageLabelList):
    def __init__(self, *args, **kwargs):
        super(TestImageBBoxLabelList, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('data/coco_instance/')
        self.xml_path = osp.join(self.data_path, "detection_xml")
        self.All_NUMBER = 10
        self.ALL_LABELS = ["person", "dog"]
        self.PRE_IMAGE_SHAPE = (427, 640, 3)
        self.image_list = [osp.normpath(x) for x in data["img_path_list"]]
        self.one_image_path = osp.normpath('data/coco_instance/000000329219.jpg')
        self.label_list = data["label_list"]
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
        self.filtered_label_list = self.filtered_label_list = [{'labels': ['dog', 'person', 'person', 'person',
                                                                           'person', 'person', 'person', 'person',
                                                                           'person', 'person', 'person'],
                                                                'bboxes': [[427, 77, 616, 363], [403, 36, 463, 311],
                                                                           [260, 0, 414, 336], [202, 1, 303, 279],
                                                                           [155, 1, 251, 273], [19, 2, 109, 257],
                                                                           [0, 2, 38, 251], [107, 0, 144, 112],
                                                                           [570, 38, 599, 66], [531, 42, 551, 99],
                                                                           [472, 53, 497, 77]]}]

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
        self._cls_ = ImageBBoxLabelList

    def test_find_idxs(self):
        ibll = self._cls_.from_folder(self.data_path)
        finded_idxs = ibll.find_idxs()
        self.assertEqual(len(finded_idxs), 0)
        ibll = self._cls_(self.image_list, self.label_list)
        finded_idxs = ibll.find_idxs()
        self.assertEqual(len(finded_idxs), self.All_NUMBER)

    @unittest.skip("skip test_databunch")
    def test_databunch(self):
        pass

    @unittest.skip("skip test_get_main_labels")
    def test_get_main_labels(self):
        pass

    @unittest.skip("skip test_resample")
    def test_resample(self):
        pass

    def test_label_from_func(self):
        ibll = self._cls_.from_folder(self.data_path)
        ibll.label_from_func(lambda x: {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]})
        self.assertJsonEqual(ibll.y, [
            {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}] * self.All_NUMBER)

    def test_filter(self):
        ibll_raw = self._cls_(self.image_list, self.label_list)

        def fun(x, y):
            return len(y["labels"]) > 10

        ibll = ibll_raw.filter(fun)
        self.assertTupleEqual(ibll.x, self.filtered_image_tuple)
        self.assertEqual(len(ibll.y[0]["labels"]), 11)

    def test_split_by_idxs(self):
        ibll = self._cls_(self.image_list, self.label_list)
        train_ibll, test_ibll = ibll.split_by_idxs(list(range(0, 5)), list(range(5, 10)))
        self.assertTupleEqual(train_ibll.x, self.split_image_tuple_train)
        self.assertTupleEqual(test_ibll.x, self.split_image_tuple_val)

    @unittest.skip("skip test_label_from_folder")
    def test_label_from_folder(self):
        pass

    def test_from_pascal_voc(self):
        ibll = self._cls_.from_pascal_voc(self.data_path, xml_dir=self.xml_path)
        self.assertCountEqual(ibll.x, self.image_list)


if __name__ == '__main__':
    unittest.main()
