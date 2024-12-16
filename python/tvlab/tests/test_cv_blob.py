from tvlab import BlobTool
import unittest
import cv2
import timeit
import torch
import numpy as np
import os.path as osp


class TestBlob(unittest.TestCase):
    def test_init(self):
        props = {'bbox_axis_aspect': [0.12021857923497267,
                1.178082191780822,
                0.7299270072992701,
                27.75,
                1.0167597765363128,
                5.514285714285714],
                'bbox_h': [549, 146, 137, 8, 179, 70],
                'bbox_w': [66, 172, 100, 222, 182, 386],
                'centroid': [(95.5, 366.0),
                (329.5, 190.4942254082039),
                (545.6733238231099, 202.60566537599348),
                (820.8759954493743, 202.5130830489192),
                (338.53198982586576, 488.03204852279396),
                (803.5, 438.5)],
                'compactness': [3.301070475834764,
                1.2611476493971623,
                3.6590227968305284,
                9.219944682573201,
                1.1067121942507345,
                2.428155458943679],
                'convex_area': [35620.0, 24794.0, 9339.0, 1535.5, 25457.0, 26565.0],
                'convexity': [1.0172375070185289,
                1.012745018956199,
                0.5254309883285149,
                1.144903940084663,
                1.0038496287857956,
                1.0171277997364954],
                'hole': [0, 0, 0, 0, 0, 0],
                # 'min_bbox_angle': [90.0, 90.0, -0.0, -0.0, 90.0, 90.0],
                'min_bbox_axis_aspect': [8.430769230769231,
                0.847953216374269,
                0.7279412276192222,
                31.57142851304034,
                0.9834254143646409,
                0.17922077922077922],
                'min_bbox_h': [65.0,
                171.0,
                135.99996948242188,
                6.999999046325684,
                181.0,
                385.0],
                'min_bbox_w': [548.0,
                145.0,
                98.99998474121094,
                220.99996948242188,
                178.0,
                69.0],
                'perimeter': [1226.0,
                630.8284270763397,
                475.0020879507065,
                451.3137083053589,
                596.156415939331,
                908.0],
                'rectangularity': [1.0172375070185289,
                1.012704174228675,
                0.3644534950544844,
                1.1363933304901945,
                0.7931901421565585,
                1.0171277997364954],
                'roundness': [0.4850996339493523,
                0.8769380840519058,
                0.7099573739970291,
                0.4301010768687906,
                0.9932866226375837,
                0.531502090316429],
                'orientation': [1.5707963267948966, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0]}
        img = cv2.imread(osp.normpath('./data/12.png'), 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        blob_tool_opencv = BlobTool(th)
        blobs_opencv = blob_tool_opencv.blobs
        for prop, value in props.items():
            blist_opencv = []
            for bbo in blobs_opencv:
                bbot = getattr(bbo, prop, None)
                if isinstance(bbot, np.ndarray):
                    bbot = bbot.tolist()
                blist_opencv.append(bbot)

            self.assertEqual(np.round(value, decimals=2).tolist(), np.round(value, decimals=2).tolist(), "error at {}".format(prop))

    def test_hole(self):
        img_path = {'./data/cv/hole_test01.png': [0, 1, 0, 0, 0, 1, 0, 0, 0],
                './data/cv/hole_test02.tif': [0, 1, 1, 0, 1],
                './data/cv/hole_test03.tif': [3, 0]}
        for path, value in img_path.items():
            img = cv2.imread(osp.normpath(path), 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

            blob_tool_opencv = BlobTool(th)
            blobs_opencv = blob_tool_opencv.blobs
            blist_opencv = []
            prop = "hole"
            for bbo in blobs_opencv:
                bbot = getattr(bbo, prop, None)
                if isinstance(bbot, np.ndarray):
                    bbot = bbot.tolist()
                blist_opencv.append(bbot)
            self.assertEqual(value, blist_opencv, "error at {}".format(path))

    def test_speedup(self):
        th = cv2.imread(osp.normpath('./data/cv/fmloc_aa_bin_4.png'), cv2.IMREAD_GRAYSCALE)

        start = timeit.default_timer()
        times = 10
        for i in range(times):
            blob_tool = BlobTool(th)
            list_area = []
            list_roundness = []
            list_compactness = []
            list_rectangularity = []
            list_min_bbox_axis_aspect = []
            list_region = []
            list_perimeter = []
            list_orientation = []
            list_moment = []
            list_contour = []
            list_centroid = []
            list_convex_area = []
            list_blob_image = []
            list_hole = []
            list_min_bbox = []
            list_bbox = []
            list_convexity = []
            list_min_bbox_w = []
            list_min_bbox_h = []
            list_min_bbox_angle = []
            list_bbox_w = []
            list_bbox_h = []
            list_bbox_axis_aspect = []
            for i, blob in enumerate(blob_tool.blobs):
                list_area.append(blob.area)
                list_blob_image.append(blob.blob_image)
                list_centroid.append(blob.centroid)
                # list_region.append(blob.get_region())
                list_hole.append(blob.hole)
                list_orientation.append(blob.orientation)

                list_convex_area.append(blob.convex_area)
                list_convexity.append(blob.convexity)

                list_contour.append(blob.contour)
                list_moment.append(blob.moments)
                list_roundness.append(blob.roundness)
                list_min_bbox.append(blob.min_bbox)
                list_min_bbox_angle.append(blob.min_bbox_angle)
                list_min_bbox_w.append(blob.min_bbox_w)
                list_min_bbox_h.append(blob.min_bbox_h)
                list_min_bbox_axis_aspect.append(blob.min_bbox_axis_aspect)
                list_rectangularity.append(blob.rectangularity)

                list_bbox.append(blob.bbox)
                list_bbox_w.append(blob.bbox_w)
                list_bbox_h.append(blob.bbox_h)
                list_bbox_axis_aspect.append(blob.bbox_axis_aspect)

                list_perimeter.append(blob.perimeter)
                list_compactness.append(blob.compactness)
        end = timeit.default_timer()
        time = (end - start) / times
        flg = False
        if time < 1.7:
            flg = True
        self.assertTrue(flg, True)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestBlob("test_init"),
        TestBlob("test_hole"),
        TestBlob("test_speedup"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
