from tvlab import *
import unittest
import shutil


class TestBasic(unittest.TestCase):

    def test_safe_div(self):
        a = safe_div(1, 0)
        self.assertEqual(a, 0)

        a = safe_div(6, 2)
        self.assertEqual(a, 3)

    def test_least_common_multiple(self):
        a = least_common_multiple(6, 9)
        self.assertEqual(a, 18)

        a = least_common_multiple(3, 3)
        self.assertEqual(a, 3)

        a = least_common_multiple(3, 9)
        self.assertEqual(a, 9)

    def test_polygon_to_bbox(self):
        box = polygon_to_bbox([1, 2, 3, 4, 2, 2, 5, 8])
        self.assertListEqual(box, [1, 2, 5, 8])

    def test_draw_bboxes_on_img(self):
        img = cv2.imread(osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg'))
        draw_img = draw_bboxes_on_img(img, [[1, 20, 50, 80]], labels=['test'])
        self.assertIsInstance(draw_img, np.ndarray)

    def test_draw_polygons_on_img_pro(self):
        img = cv2.imread(osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg'))
        draw_img = draw_polygons_on_img_pro(
            img, [[20, 20, 50, 80, 500, 500]], labels=['test'])
        self.assertIsInstance(draw_img, np.ndarray)

    def test_draw_bboxes_on_img_pro(self):
        img = cv2.imread(osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg'))
        d_img = draw_bboxes_on_img_pro(img, [[1, 20, 50, 80]], labels=['test'])
        self.assertIsInstance(d_img, np.ndarray)

    def test_obj_to_pkl(self):
        img = cv2.imread(osp.normpath('data/hymenoptera_data/bees/36900412_92b81831ad.jpg'))
        obj_to_pkl(img, osp.normpath('./pkl/test.pkl'))
        img = obj_from_pkl(osp.normpath('./pkl/test.pkl'))
        self.assertIsInstance(img, np.ndarray)
        shutil.rmtree(osp.normpath('./pkl'), ignore_errors=True)

    def test_obj_to_json(self):
        a = {'test': 1}
        obj_to_json(a, osp.normpath('./json/test.json'))
        json = obj_from_json(osp.normpath('./json/test.json'))
        self.assertEqual(len(a), len(json))
        shutil.rmtree(osp.normpath('./json'), ignore_errors=True)

    def test_obj_to_yaml(self):
        a = [1., 2, 3]
        obj_to_yaml(a, osp.normpath('./json/test.yaml'))
        b = obj_from_yaml(osp.normpath('./json/test.yaml'))
        self.assertListEqual(a, b)
        shutil.rmtree(osp.normpath('./json'), ignore_errors=True)

    @unittest.skip("Skip test_set_gpu_visible")
    def test_set_gpu_visible(self):
        set_gpu_visible(0)

    def test_set_notebook_url(self):
        set_notebook_url('172.0.0.1:8888')
        url = get_notebook_url()
        self.assertEqual('172.0.0.1:8888', url)

    def test_mask_to_polygon(self):
        mask = cv2.imread(osp.normpath('data/person_mask.png'), 0)
        pl = mask_to_polygon(mask)
        self.assertIsInstance(pl, list)
        self.assertEqual(len(pl), 3080)

    def test_mask_to_polygons(self):
        mask = cv2.imread(osp.normpath('data/person_mask.png'), 0)
        pls = mask_to_polygons(mask)
        self.assertIsInstance(pls, list)
        self.assertEqual(len(pls), 2)


if __name__ == '__main__':
    unittest.main()
