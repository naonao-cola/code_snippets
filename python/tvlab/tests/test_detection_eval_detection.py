'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import unittest
from tvlab import *
import os


class TestEvalDetection(unittest.TestCase):
    def test_get_error_images(self):
        y_true = BBoxLabelList([
            {'labels': [], 'bboxes': []},
            {'labels': ['A'], 'bboxes': [[100, 100, 200, 200]]},
            {'labels': ['A', 'B', 'A'],
             'bboxes': [[100, 130, 200, 250],
                        [200, 300, 300, 400],
                        [400, 500, 500, 600]]}])

        y_pred_full = BBoxLabelList([
            {'labels': [], 'bboxes': []},
            {'labels': ['A'], 'bboxes': [[100, 100, 200, 200, 0.8]]},
            {'labels': ['A', 'B', 'A'],
             'bboxes': [[100, 130, 200, 250, 0.9],
                        [200, 300, 300, 400, 1.0],
                        [400, 500, 500, 600, 2.0]]}])
        evad = EvalDetection(y_pred=y_pred_full, y_true=y_true, iou_threshold=0.3)
        err_idx, err_pred_idx, err_targ_idx = evad.get_error_images()
        self.assertEqual(err_idx, [])
        self.assertEqual(err_pred_idx, [])
        self.assertEqual(err_targ_idx, [])

        y_pred_none = BBoxLabelList([
            {'labels': [], 'bboxes': []},
            {'labels': [], 'bboxes': []},
            {'labels': [], 'bboxes': []}])
        evad = EvalDetection(y_pred=y_pred_none, y_true=y_true, iou_threshold=0.3)
        err_idx, err_pred_idx, err_targ_idx = evad.get_error_images()
        self.assertEqual(err_idx, [1, 2])
        self.assertEqual(err_pred_idx, [[], []])
        self.assertEqual(err_targ_idx, [[0], [0, 2, 1]])

        evad = EvalDetection(y_pred=y_pred_full, y_true=y_pred_none, iou_threshold=0.3)
        err_idx, err_pred_idx, err_targ_idx = evad.get_error_images()
        self.assertEqual(err_idx, [1, 2])
        self.assertEqual(err_pred_idx, [[0], [0, 2, 1]])
        self.assertEqual(err_targ_idx, [[], []])


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestEvalDetection("test_get_error_images"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
