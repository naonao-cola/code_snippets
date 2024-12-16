from tvlab import *
import unittest
import numpy as np


class TestBboxOverlaps(unittest.TestCase):
    def test_bbox_overlaps(self):
        box1 = np.array([[0, 0, 1, 1]])
        box2 = np.array([[0, 0, 1, 1]])
        iou = bbox_overlaps(box1, box2)
        self.assertEqual(iou, [[1.]])

        box2 = np.array([[-2, -2, -1, -1]])
        iou = bbox_overlaps(box1, box2)
        self.assertEqual(iou, [[0.]])

        box2 = np.array([[0.5, 0.5, 1, 1]])
        iou = bbox_overlaps(box1, box2)
        flag = True
        if np.abs(iou - 0.5625) > 0.01:
            flag = False
        self.assertTrue(flag, True)

    def test_nms(self):
        bboxes = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [
            1, 1, 2, 2], [-2, -2, -1, -1]])
        res = nms(bboxes)
        self.assertEqual(len(res), 3)

    def test_y_nms(self):
        y = {'labels': ['A', 'B'], 'bboxes': [
            [10, 20, 100, 200], [20, 40, 50, 80]]}
        res = y_nms(y)
        self.assertEqual(len(res), 2)


if __name__ == '__main__':
    unittest.main()
