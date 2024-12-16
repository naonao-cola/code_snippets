from tvlab import TemplateBasedMatching
import unittest
import os
import cv2
import shutil
import os.path as osp
import time


class TestTemplateBasedMatching(unittest.TestCase):
    root_dir = osp.normpath('./data/cv/')
    template_dir1 = os.path.join(root_dir, 'white_luosi_ok1_template.png')
    img_ok_dir1 = os.path.join(root_dir, 'white_luosi_ok2.png')
    img_ng_dir1 = os.path.join(root_dir, 'white_luosi_ng4.png')
    template_dir2 = os.path.join(root_dir, 'blue_ok_1_template.png')

    yaml_dir = os.path.join(root_dir, 'ec.yaml')
    shutil.rmtree(yaml_dir, ignore_errors=True)

    def test_init(self):
        tbm = TemplateBasedMatching()
        self.assertIsInstance(tbm.templates, dict)

    def test_add(self):
        tbm = TemplateBasedMatching()
        self.assertEqual(len(tbm.templates), 0)

        template1 = cv2.imread(self.template_dir1, 0)
        tbm.add(template=template1, class_id='white')
        self.assertEqual(len(tbm.templates), 1)
        self.assertEqual(len(tbm.templates['white'][0].shape), 2)
        self.assertEqual(tbm.templates['white'][1], cv2.TM_CCOEFF_NORMED)

        template2 = cv2.imread(self.template_dir2, 0)
        tbm.add(template=template2, class_id='blue')
        self.assertEqual(len(tbm.templates), 2)

    def test_save(self):
        tbm = TemplateBasedMatching()
        template1 = cv2.imread(self.template_dir1, 0)
        tbm.add(template=template1, class_id='white')
        template2 = cv2.imread(self.template_dir2, 0)
        tbm.add(template=template2, class_id='blue')
        tbm.save(self.yaml_dir)

        self.assertTrue(os.path.exists(self.yaml_dir))

    def test_load(self):
        tbm = TemplateBasedMatching()
        tbm.load(self.yaml_dir)
        self.assertEqual(len(tbm.templates), 2)

    def test_find(self):
        tbm = TemplateBasedMatching()
        tbm.load(self.yaml_dir)

        # 1. test ok sample
        # too high threshold
        img = cv2.imread(self.img_ok_dir1, 0)
        res = tbm.find(img)
        self.assertEqual(len(res['white']), 0)
        self.assertEqual(len(res['blue']), 0)

        # turn down threshold
        res = tbm.find(img, score_threshold=80)
        self.assertEqual(len(res['white']), 1)
        self.assertEqual(len(res['blue']), 0)

        # move on
        res = tbm.find(img, score_threshold=20)
        self.assertEqual(len(res['white']), 9)
        self.assertEqual(len(res['blue']), 14)

        # use iou
        res = tbm.find(img, score_threshold=20, iou_threshold=0.2)
        self.assertEqual(len(res['white']), 5)
        self.assertEqual(len(res['blue']), 6)

        # choose top 1
        res = tbm.find(img, score_threshold=20, topk=1)
        self.assertEqual(len(res['white']), 1)
        self.assertEqual(len(res['blue']), 1)

        # test time consuming
        start = time.time()
        res = tbm.find(img, class_ids=['white'], score_threshold=20, topk=1)
        spend_time = time.time() - start
        time_flag = True
        if spend_time > 0.05:
            time_flag = False
        self.assertTrue(time_flag, True)

        flag = 'blue' in res
        self.assertEqual(flag, False)

        # 2. test ng sample
        img = cv2.imread(self.img_ng_dir1, 0)
        res = tbm.find(img, score_threshold=80)
        self.assertEqual(len(res['white']), 0)
        self.assertEqual(len(res['blue']), 0)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTemplateBasedMatching("test_init"),
        TestTemplateBasedMatching("test_add"),
        TestTemplateBasedMatching("test_save"),
        TestTemplateBasedMatching("test_load"),
        TestTemplateBasedMatching("test_find")
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
