import unittest
from tvlab import *
from imgaug import augmenters as iaa
import cv2
import numpy as numpy
import mimetypes
import os
import shutil


def resize_256(img):
    img = cv2.resize(img, (256, 256))
    return img


def center_crop(img):
    return iaa.CropToFixedSize(224, 224, position='center').augment_image(img)


def random_crop(img):
    return iaa.CropToFixedSize(224, 224).augment_image(img)


def cmp_json(src_data, dst_data):
    if isinstance(src_data, dict):
        if dst_data is None:
            return False

        """若为dict格式"""
        for key in dst_data:
            if key not in src_data:
                return False
        for key in src_data:
            if key in dst_data:
                """递归"""
                flag = cmp_json(src_data[key], dst_data[key])
                if not flag:
                    return False
            else:
                return False
        return True
    elif isinstance(src_data, list):
        if dst_data is None:
            return False

        """若为list格式"""
        if len(src_data) != len(dst_data):
            return False
        for src_list, dst_list in zip(src_data, dst_data):
            """递归"""
            flag = cmp_json(src_list, dst_list)
            if not flag:
                return False
        return True
    else:
        if str(src_data) != str(dst_data):
            return False
        else:
            return True


class TestImageMultiLabelList(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageMultiLabelList, self).__init__(*args, **kwargs)
        self._cls_ = ImageMultiLabelList
        self.data_path = osp.normpath('data/defect_data/defect_image')
        self.PRE_IMAGE_SHAPE = (1024, 1280, 3)
        self.main_label_list = ['ACT', 'ACT', 'ACT',
                                'ACT', 'GA1', 'GA1', 'GA2', 'GA2', 'SDT', 'SDT']
        self.ALL_LABELS = [['ACT', 'GA1', 'GA2', 'SDT'], [
            'D8D', 'H2U', 'L2T', 'P6U', 'R1D', 'R2D', 'R3T']]
        self.All_NUMBER = 10
        self.one_image_path = osp.normpath(
            'data/defect_data/defect_image/ACT_CHOPIN_120_0453701589__R2D_956_447_1040_531b.jpg')
        self.main_labelset = ['ACT', 'GA1', 'GA2', 'SDT']
        self.image_list = ['data/defect_data/defect_image/ACT_CHOPIN_120_0453701589__R2D_956_447_1040_531.jpg',
                           'data/defect_data/defect_image/ACT_CHOPIN_120_0453701589__R2D_956_447_1040_531b.jpg',
                           'data/defect_data/defect_image/ACT_MOZART_100_0619183668__D8D_757_836_780_859.jpg',
                           'data/defect_data/defect_image/ACT_MOZART_100_0619183668__D8D_757_836_780_859b.jpg',
                           'data/defect_data/defect_image/GA1_CHOPIN_100_1707181449__R1D_698_533_725_560.jpg',
                           'data/defect_data/defect_image/GA1_MOZART_100_1779594497__D8D_451_552_483_584.jpg',
                           'data/defect_data/defect_image/GA2_CHOPIN_100_0945526972__L2T_357_167_866_676.jpg',
                           'data/defect_data/defect_image/GA2_MOZART_100_0859707253__P6U_176_0_1195_980.jpg',
                           'data/defect_data/defect_image/SDT_CHOPIN_080_0035449223__H2U_408_568_422_582.jpg',
                           'data/defect_data/defect_image/SDT_MOZART_080_0009418307__R3T_447_107_579_239.jpg']
        self.image_list = [osp.normpath(x) for x in self.image_list]
        self.label_list = [('ACT', 'R2D'), ('ACT', 'R2D'), ('ACT', 'D8D'), ('ACT', 'D8D'), (
            'GA1', 'R1D'), ('GA1', 'D8D'), ('GA2', 'L2T'), ('GA2', 'P6U'), ('SDT', 'H2U'), ('SDT', 'R3T')]
        self.none_label_list = [None] * 10

    def assertJsonEqual(self, j1, j2):
        self.assertTrue(cmp_json(j1, j2))

    def assertJsonNotEqual(self, j1, j2):
        self.assertFalse(cmp_json(j1, j2))

    def test_ImageMultiLabelList_init_without_label(self):
        ill = self._cls_(self.image_list)
        y_cls = ill.y.__class__
        self.assertEqual(len(ill), self.All_NUMBER)
        self.assertJsonEqual(ill.x, self.image_list)
        self.assertJsonEqual(ill.y, y_cls(self.none_label_list))

    def test_set_img_mode(self):
        ill = self._cls_(self.image_list)
        self.assertEqual(ill.img_mode, "RGB")
        ill.set_img_mode("L")
        self.assertEqual(ill.img_mode, "L")

    def test_cache_images(self):
        ill = self._cls_(self.image_list)
        ill = ill.cache_images(workers=5)
        self.assertEqual(len(ill.cache_img), self.All_NUMBER)
        self.assertListEqual(ill.cache_x, self.image_list)

    def test_copy(self):
        ill = self._cls_(self.image_list)
        newill = ill.copy()
        self.assertEqual(ill.img_mode, newill.img_mode)
        self.assertEqual(ill.cache_x, newill.cache_x)
        self.assertEqual(ill.cache_img, newill.cache_img)
        self.assertListEqual(ill.x, newill.x)
        self.assertJsonEqual(ill.y, newill.y)

    def test_from_ill(self):
        ill = self._cls_(self.image_list)
        newill = self._cls_.from_ill(ill)
        self.assertEqual(ill.img_mode, newill.img_mode)
        self.assertEqual(ill.cache_x, newill.cache_x)
        self.assertEqual(ill.cache_img, newill.cache_img)
        self.assertListEqual(ill.x, newill.x)
        self.assertJsonEqual(ill.y, newill.y)

    def test_from_memory(self):
        img = cv2.imread(self.one_image_path)
        ill = self._cls_.from_memory([img])
        self.assertEqual(len(ill.x), 1)
        self.assertEqual(len(ill.cache_x), 1)
        self.assertEqual(len(ill.cache_img), 1)

    def test_from_folder(self):
        ill = self._cls_.from_folder(self.data_path)
        y_cls = ill.y.__class__
        self.assertEqual(len(ill), self.All_NUMBER)
        self.assertCountEqual(ill.x, self.image_list)
        self.assertJsonEqual(ill.y, y_cls(self.none_label_list))

    def test_shuffle(self):
        ill = self._cls_(self.image_list.copy(), self.label_list.copy())
        ill = ill.shuffle()
        self.assertCountEqual(ill.x, self.image_list)
        self.assertCountEqual(ill.y, self.label_list)
        self.assertJsonNotEqual(ill.x, self.image_list)
        self.assertJsonNotEqual(ill.y, self.label_list)

    def test_label_from_func(self):
        ill = self._cls_.from_folder(self.data_path)
        ill.label_from_func(lambda x: "ok")
        self.assertJsonEqual(ill.y, ["ok"] * self.All_NUMBER)

    @unittest.skip("skip test_label_from_folder")
    def test_label_from_folder(self):
        ill = self._cls_.from_folder(self.data_path)
        self.assertJsonEqual(ill.y, self.none_label_list)
        ill = ill.label_from_folder()
        ill_y = [ill.y[ill.x.index(x)] for x in self.image_list]
        self.assertJsonEqual(ill_y, self.label_list)

    @unittest.skip("skip test_sorted")
    def test_sorted(self):
        ill = self._cls_(self.image_list, self.label_list)
        ill = ill.sorted()
        self.assertTupleEqual(ill.x, self.sorted_image_tuple)

    def test_find_idxs(self):
        ill = self._cls_.from_folder(self.data_path)
        finded_idxs = ill.find_idxs()
        self.assertEqual(len(finded_idxs), 0)
        ill = ill.label_from_func(func=lambda o: (o.split(osp.sep))[-2])
        finded_idxs = ill.find_idxs()
        self.assertEqual(len(finded_idxs), self.All_NUMBER)

    def test_filter(self):
        ill = self._cls_(self.image_list, self.label_list)
        ill = ill.filter(lambda x, y: y[0] == "ACT")
        self.assertEqual(len(ill), 4)

    @unittest.skip("skip test_filter_invalid_img")
    def test_filter_invalid_img(self):
        pass

    @unittest.skip("skip test_filter_similar_img")
    def test_filter_similar_img(self):
        pass

    @unittest.skip("skip test_resample")
    def test_resample(self):
        pass

    def test_set_tfms(self):
        ill = self._cls_(self.image_list, self.label_list)
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 0)
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

        ill.set_tfms([resize_256, center_crop])
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 2)
        self.assertTupleEqual(image.shape, (224, 224, 3))

    def test_add_tfm(self):
        ill = self._cls_(self.image_list, self.label_list)
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 0)
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

        ill.add_tfm(resize_256)
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 1)
        self.assertTupleEqual(image.shape, (256, 256, 3))

        ill.add_tfm(center_crop)
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 2)
        self.assertTupleEqual(image.shape, (224, 224, 3))

    def test_clear_tfms(self):
        ill = self._cls_(self.image_list, self.label_list)
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 0)
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

        ill.set_tfms([resize_256, center_crop])
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 2)
        self.assertTupleEqual(image.shape, (224, 224, 3))

        ill.clear_tfms()
        image, _ = ill[0]
        self.assertEqual(len(ill._tfms), 0)
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

    def test_do_tfms(self):
        ill = self._cls_(self.image_list, self.label_list)
        image, _ = ill[0]
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

        ill.add_tfm(resize_256)
        image, _ = ill[0]
        self.assertTupleEqual(image.shape, (256, 256, 3))

    def test_split_by_idxs(self):
        ill = self._cls_(self.image_list, self.label_list)
        train_ill, test_ill = ill.split_by_idxs(
            list(range(0, 5)), list(range(5, 10)))
        self.assertEqual(len(train_ill), 5)
        self.assertEqual(len(test_ill), 5)

    def test_labelset(self):
        ill = self._cls_(self.image_list, self.label_list)
        self.assertCountEqual(ill.labelset(), self.ALL_LABELS)

    def test_get_main_labels(self):
        ill = self._cls_(self.image_list.copy(), self.label_list.copy())
        self.assertListEqual(ill.get_main_labels(), self.main_label_list)

    def test_get_main_labelset(self):
        ill = self._cls_(self.image_list, self.label_list)
        self.assertCountEqual(ill.get_main_labelset(), self.main_labelset)

    @unittest.skip("skip test_show_split")
    def test_show_split(self):
        pass

    @unittest.skip("skip test_split")
    def test_split(self):
        pass

    @unittest.skip("skip test_kfold")
    def test_kfold(self):
        pass

    def test_merge(self):
        train_ill = self._cls_(
            self.image_list[0:5].copy(), self.label_list[0:5].copy())
        test_ill = self._cls_(
            self.image_list[5:10].copy(), self.label_list[5:10].copy())
        ill = self._cls_.merge(train_ill, test_ill)
        self.assertListEqual(ill.x, self.image_list)
        self.assertJsonEqual(ill.y, self.label_list)
        self.assertListEqual(ill._train_idx, list(range(0, 5)))
        self.assertListEqual(ill._valid_idx, list(range(5, 10)))

    @unittest.skip("skip test_export")
    def test_export(self):
        pass

    @unittest.skip("skip test_load_img_cache")
    def test_load_img_cache(self):
        pass

    @unittest.skip("skip test_load")
    def test_load(self):
        pass

    def test_load_image(self):
        ill = self._cls_(self.image_list, self.label_list)
        image = ill.load_image(ill.x[0])
        self.assertIsInstance(image, np.ndarray)
        self.assertTupleEqual(image.shape, self.PRE_IMAGE_SHAPE)

    def test_getitem(self):
        ill = self._cls_(self.image_list.copy(), self.label_list.copy())
        for i in range(len(ill)):
            with self.subTest(i=i):
                image, label = ill[i]
                self.assertIsInstance(image, np.ndarray)
                self.assertEqual(label, self.label_list[i])

    @unittest.skip("skip test_show_sample")
    def test_show_sample(self):
        pass

    @unittest.skip("skip test_show_dist")
    def test_show_dist(self):
        pass


if __name__ == '__main__':
    unittest.main()
