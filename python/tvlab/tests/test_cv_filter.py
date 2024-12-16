from tvlab import *
import unittest
import cv2
import timeit
import torch
import numpy as np


class TestFilter(unittest.TestCase):
    def test_gaussian_blur(self):
        img_batch = [[[[-0.1677, -0.1765],
                       [0.2138, 1.5242]],

                      [[-0.9688, 1.3151],
                       [0.4584, -0.7818]],

                      [[-0.1740, -0.7023],
                       [0.5339, -0.3878]]]]
        img_batch = np.array(img_batch)
        img_batch = torch.from_numpy(img_batch)
        img_batch = torch.tensor(img_batch, dtype=torch.float32)
        re = gaussian_blur(img_batch)
        re_t = 'tensor(0.0674)'
        self.assertEqual(str(re[0][0][0][0]), re_t)

    def test_median_blur(self):
        img_batch = [[[[-0.1677, -0.1765],
                       [0.2138, 1.5242]],

                      [[-0.9688, 1.3151],
                       [0.4584, -0.7818]],

                      [[-0.1740, -0.7023],
                       [0.5339, -0.3878]]]]
        img_batch = np.array(img_batch)
        img_batch = torch.from_numpy(img_batch)
        img_batch = torch.tensor(img_batch, dtype=torch.float32)
        re = median_blur(img_batch)
        re_t = [[[[0., 0.],
                  [0., 0.]],

                 [[0., 0.],
                  [0., 0.]],

                 [[0., 0.],
                  [0., 0.]]]]
        re = re.numpy()
        re = re.tolist()
        self.assertEqual(re, re_t)

    def test_peak_local_max(self):
        t = [[[-0.1677, -0.1765],
              [0.2138, 1.5242]]]
        t = torch.tensor(torch.from_numpy(np.array(t)), dtype=torch.float32)
        re1 = peak_local_max(t, indices=True, num_peaks=2)
        re2 = peak_local_max(t, indices=False, num_peaks=2).numpy().tolist()
        re1_t = [[(1, 1)]]
        re2_t = [[[False, False], [False, True]]]
        flg = False
        if re1 == re1_t and re2 == re2_t:
            flg = True
        self.assertTrue(flg, True)


if __name__ == '__main__':
    unittest.main()
