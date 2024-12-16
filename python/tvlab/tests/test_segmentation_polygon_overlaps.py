from tvlab import *
import unittest


class TestPolygonOverlaps(unittest.TestCase):
    def test_polygon_overlaps(self):
        p1 = [[1, 1, 2, 2, 3, 1]]
        p2 = [[1, 1, 2, 2, 3, 1]]
        iou = polygon_overlaps(p1, p2)
        self.assertEqual(len(iou), 1)
        self.assertEqual(iou, 1)

        p1 = [[1, 1, 2, 2, 3, 1]]
        p2 = [[0, 0, -1, -1, -3, -3]]
        iou = polygon_overlaps(p1, p2)
        self.assertEqual(iou, 0)

        p1 = [[1, 1, 2, 2, 3, 1]]
        p2 = [[0.5, 0.5, 2, 2, 3, 1]]
        iou = polygon_overlaps(p1, p2)
        flag = True
        if np.abs(0.6666667 - iou[0]) > 0.001:
            flag = False
        self.assertTrue(flag, True)

    def test_polygon_nms(self):
        ps = [[1, 1, 2, 2, 3, 1], [1, 1, 2, 2, 3, 1], [1, 1, 0, 2, 3, 1]]
        res = polygon_nms(ps)
        self.assertEqual(len(res), 1)


if __name__ == '__main__':
    unittest.main()
