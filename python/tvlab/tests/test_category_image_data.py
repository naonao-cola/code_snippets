'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import unittest
from tvlab import *
from imgaug import augmenters as iaa
import cv2
import numpy as np
import mimetypes


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


class TestImageLabelList(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestImageLabelList, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('data/hymenoptera_data/')
        self.All_NUMBER = 10
        self.ALL_LABELS = ["ants", "bees"]
        self.PRE_IMAGE_SHAPE = (173, 500, 3)
        self.image_list = ['data/hymenoptera_data/bees/36900412_92b81831ad.jpg',
                           'data/hymenoptera_data/bees/196658222_3fffd79c67.jpg',
                           'data/hymenoptera_data/bees/2452236943_255bfd9e58.jpg',
                           'data/hymenoptera_data/bees/1092977343_cb42b38d62.jpg',
                           'data/hymenoptera_data/bees/2634617358_f32fd16bea.jpg',
                           'data/hymenoptera_data/ants/342438950_a3da61deab.jpg',
                           'data/hymenoptera_data/ants/0013035.jpg',
                           'data/hymenoptera_data/ants/20935278_9190345f6b.jpg',
                           'data/hymenoptera_data/ants/154124431_65460430f2.jpg',
                           'data/hymenoptera_data/ants/5650366_e22b7e1065.jpg']
        self.image_list = [osp.normpath(x) for x in self.image_list]
        self.one_image_path = osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg')
        self.label_list = ["bees"] * 5 + ["ants"] * 5
        self.none_label_list = [None] * 10
        self.sorted_image_tuple = ('data/hymenoptera_data/ants/0013035.jpg',
                                   'data/hymenoptera_data/bees/1092977343_cb42b38d62.jpg',
                                   'data/hymenoptera_data/ants/154124431_65460430f2.jpg',
                                   'data/hymenoptera_data/bees/196658222_3fffd79c67.jpg',
                                   'data/hymenoptera_data/ants/20935278_9190345f6b.jpg',
                                   'data/hymenoptera_data/bees/2452236943_255bfd9e58.jpg',
                                   'data/hymenoptera_data/bees/2634617358_f32fd16bea.jpg',
                                   'data/hymenoptera_data/ants/342438950_a3da61deab.jpg',
                                   'data/hymenoptera_data/bees/36900412_92b81831ad.jpg',
                                   'data/hymenoptera_data/ants/5650366_e22b7e1065.jpg')
        self.sorted_image_tuple = tuple([osp.normpath(x) for x in self.sorted_image_tuple])
        self.sorted_label_list = ['ants', 'bees', 'ants', 'bees', 'ants', 'bees', 'bees', 'ants', 'bees', 'ants']
        self.ants_image_tuple = ('data/hymenoptera_data/ants/342438950_a3da61deab.jpg',
                                 'data/hymenoptera_data/ants/0013035.jpg',
                                 'data/hymenoptera_data/ants/20935278_9190345f6b.jpg',
                                 'data/hymenoptera_data/ants/154124431_65460430f2.jpg',
                                 'data/hymenoptera_data/ants/5650366_e22b7e1065.jpg')
        self.ants_image_tuple = tuple([osp.normpath(x) for x in self.ants_image_tuple])
        self.ants_label_list = ["ants"] * 5
        self.bees_image_tuple = ('data/hymenoptera_data/bees/36900412_92b81831ad.jpg',
                                 'data/hymenoptera_data/bees/196658222_3fffd79c67.jpg',
                                 'data/hymenoptera_data/bees/2452236943_255bfd9e58.jpg',
                                 'data/hymenoptera_data/bees/1092977343_cb42b38d62.jpg',
                                 'data/hymenoptera_data/bees/2634617358_f32fd16bea.jpg',)
        self.bees_image_tuple = tuple([osp.normpath(x) for x in self.bees_image_tuple])
        self.bees_label_list = ["bees"] * 5
        self._cls_ = ImageLabelList

    def assertJsonEqual(self, j1, j2):
        self.assertTrue(cmp_json(j1, j2))

    def assertJsonNotEqual(self, j1, j2):
        self.assertFalse(cmp_json(j1, j2))

    def test_ImageLabelList_init_without_label(self):
        ill = self._cls_(self.image_list)
        y_cls = ill.y.__class__
        self.assertEqual(len(ill), self.All_NUMBER)
        self.assertJsonEqual(ill.x, self.image_list)
        self.assertJsonEqual(ill.y, y_cls(self.none_label_list))

    def test_ImageLabelList_init_with_label(self):
        ill = self._cls_(self.image_list, self.label_list)
        self.assertEqual(len(ill), self.All_NUMBER)
        self.assertListEqual(ill.x, self.image_list)
        self.assertJsonEqual(ill.y, self.label_list)

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
        img = open_image(self.one_image_path)
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

    @unittest.skip("Skip test_from_label_info")
    def test_from_label_info(self):
        pass

    @unittest.skip("Skip test_to_label_info")
    def test_to_label_info(self):
        pass

    @unittest.skip("Skip test_databunch")
    def test_databunch(self):
        pass

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

    def test_label_from_folder(self):
        ill = self._cls_.from_folder(self.data_path)
        self.assertJsonEqual(ill.y, self.none_label_list)
        ill = ill.label_from_folder()
        ill_y = [ill.y[ill.x.index(x)] for x in self.image_list]
        self.assertJsonEqual(ill_y, self.label_list)

    def test_sorted(self):
        ill = self._cls_(self.image_list, self.label_list)
        ill = ill.sorted()
        self.assertTupleEqual(ill.x, self.sorted_image_tuple)

    def test_find_idxs(self):
        ill = self._cls_.from_folder(self.data_path)
        finded_idxs = ill.find_idxs()
        self.assertEqual(len(finded_idxs), 0)
        ill = ill.label_from_folder()
        finded_idxs = ill.find_idxs()
        self.assertEqual(len(finded_idxs), self.All_NUMBER)

    def test_filter(self):
        ill = self._cls_(self.image_list, self.label_list)
        ill = ill.filter(lambda x, y: y == "ants")
        self.assertTupleEqual(ill.x, self.ants_image_tuple)
        self.assertJsonEqual(ill.y, self.ants_label_list)

    def test_filter_invalid_img(self):
        ill = self._cls_(self.image_list, self.label_list)
        fil_img = ill.filter_invalid_img()
        self.assertEqual(len(fil_img), 10)

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
        train_ill, test_ill = ill.split_by_idxs(list(range(0, 5)), list(range(5, 10)))
        self.assertTupleEqual(train_ill.x, self.bees_image_tuple)
        self.assertJsonEqual(train_ill.y, self.bees_label_list)
        self.assertTupleEqual(test_ill.x, self.ants_image_tuple)
        self.assertJsonEqual(test_ill.y, self.ants_label_list)

    def test_labelset(self):
        ill = self._cls_(self.image_list, self.label_list)
        self.assertCountEqual(ill.labelset(), self.ALL_LABELS)

    def test_get_main_labels(self):
        ill = self._cls_(self.image_list.copy(), self.label_list.copy())
        self.assertListEqual(ill.get_main_labels(), self.label_list)

    def test_get_main_labelset(self):
        ill = self._cls_(self.image_list, self.label_list)
        self.assertCountEqual(ill.get_main_labelset(), self.ALL_LABELS)

    @unittest.skip("skip test_show_split")
    def test_show_split(self):
        pass

    @unittest.skip("skip test_split")
    def test_split(self):
        pass

    def test_kfold(self):
        ill = self._cls_(self.image_list, self.label_list)
        k_folder_ill = ill.kfold()
        self.assertEqual(len(k_folder_ill), 5)

    def test_merge(self):
        train_ill = self._cls_(self.image_list[0:5].copy(), self.label_list[0:5].copy())
        test_ill = self._cls_(self.image_list[5:10].copy(), self.label_list[5:10].copy())
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

    def test_from_and_to_turbox_data(self):
        ill = self._cls_(self.image_list, self.label_list)
        turbox_data = ill.to_turbox_data()
        tu_ill = self._cls_.from_turbox_data(turbox_data)
        self.assertEqual(len(tu_ill), 10)


class TestCategoryImageDataFucs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCategoryImageDataFucs, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('data/hymenoptera_data/')
        self.one_image_path = osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg')
        self.IMAGE_EXTENSIONS = set(k for k, v in mimetypes.types_map.items() if v.startswith('image'))
        self.one_image_tuple = (500, 173)

    def test_get_files(self):
        res = get_files(self.data_path)
        self.assertEqual(len(res), 2)
        res = get_files(self.data_path, extensions=self.IMAGE_EXTENSIONS)
        self.assertEqual(len(res), 0)
        res = get_files(self.data_path, extensions=None, recurse=True)
        self.assertEqual(len(res), 10)
        res = get_files(self.data_path, extensions=self.IMAGE_EXTENSIONS, recurse=True)
        self.assertEqual(len(res), 10)

    def test_get_image_files(self):
        res = get_image_files(self.data_path, check_ext=False, recurse=False)
        self.assertEqual(len(res), 2)
        res = get_image_files(self.data_path, check_ext=True, recurse=False)
        self.assertEqual(len(res), 0)
        res = get_image_files(self.data_path, check_ext=False, recurse=True)
        self.assertEqual(len(res), 10)
        res = get_image_files(self.data_path, check_ext=True, recurse=True)
        self.assertEqual(len(res), 10)

    def test_get_image_res(self):
        self.assertTupleEqual(get_image_res(self.one_image_path), self.one_image_tuple)

    def test_open_image(self):
        image = open_image(self.one_image_path)
        self.assertIsInstance(image, np.ndarray)
        self.assertTupleEqual(image.shape, (173, 500, 3))

    @unittest.skip("skip test_save_image")
    def test_save_image(self):
        pass


@unittest.skip("skip TestZipImageLabelList")
class TestZipImageLabelList(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestZipImageLabelList, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('../data/hymenoptera_data/')
        self.image_list = ['data/hymenoptera_data/bees/36900412_92b81831ad.jpg',
                           'data/hymenoptera_data/bees/196658222_3fffd79c67.jpg',
                           'data/hymenoptera_data/bees/2452236943_255bfd9e58.jpg',
                           'data/hymenoptera_data/bees/1092977343_cb42b38d62.jpg',
                           'data/hymenoptera_data/bees/2634617358_f32fd16bea.jpg',
                           'data/hymenoptera_data/ants/342438950_a3da61deab.jpg',
                           'data/hymenoptera_data/ants/0013035.jpg',
                           'data/hymenoptera_data/ants/20935278_9190345f6b.jpg',
                           'data/hymenoptera_data/ants/154124431_65460430f2.jpg',
                           'data/hymenoptera_data/ants/5650366_e22b7e1065.jpg']
        self.image_list = [osp.normpath(x) for x in self.image_list]
        self.label_list = ["bees"] * 5 + ["ants"] * 5


if __name__ == '__main__':
    unittest.main()
