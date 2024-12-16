from tvlab import *
import unittest
import cv2
import timeit
import torch


class TestBasic(unittest.TestCase):
    def test_ncc(self):
        M = 2
        N = 4
        K = 5
        x = torch.randn(M, K)
        y = torch.randn(N, K)
        re = ncc(x, y)
        self.assertEqual(re.shape[0], M)
        self.assertEqual(re.shape[1], N)


if __name__ == '__main__':
    unittest.main()
