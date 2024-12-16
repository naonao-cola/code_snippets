import unittest
import math
from functools import partial
import cv2
from tvlab import BlobTool
import os.path as osp


class TestBlobTool(unittest.TestCase):

    def test_blob_union(self):
        binary_img = cv2.imread(
            osp.normpath('data/cv/binary-blob-new_wm.png'), cv2.IMREAD_GRAYSCALE)
        blobTool = BlobTool(binary_img)
        old_blobs = blobTool.blobs

        def func(blob1, blob2, orientation_thresh):
            orientation1 = blob1.orientation
            orientation2 = blob2.orientation

            orientation_dist = abs(orientation1 - orientation2)
            if orientation_dist > math.pi / 2:
                orientation_dist = math.pi - \
                                   abs(orientation1) - abs(orientation2)

            if orientation_dist < orientation_thresh:
                return True
            return False

        wrapper = partial(func, orientation_thresh=0.03)
        blobTool.union(wrapper)
        newblobs = blobTool.blobs

        self.assertEqual(newblobs[3].area, 1039)
        self.assertEqual(len(newblobs), 6)
        self.assertEqual(newblobs[3].area,
                         old_blobs[4].area + old_blobs[3].area)

    def test_blob_filter(self):
        binary_img = cv2.imread(
            osp.normpath('data/cv/binary_blob_wm.png'), cv2.IMREAD_GRAYSCALE)
        blobTool = BlobTool(binary_img)
        blobs = blobTool.blobs

        blobTool.filter(lambda blob: blob.area > 520)
        blobs_filtered = blobTool.blobs
        self.assertEqual(5, len(blobs_filtered))
        self.assertGreater(len(blobs), len(blobs_filtered))

    def test_blob_union_by_dist(self):
        binary_img = cv2.imread(
            osp.normpath('data/cv/binary_blob_wm.png'), cv2.IMREAD_GRAYSCALE)
        print(binary_img.shape)
        blobTool = BlobTool(binary_img)
        blobTool.union_by_dist(30)
        blobs = blobTool.blobs
        self.assertEqual(3, len(blobs))

    def test_blob_union_by_orientation(self):
        binary_img = cv2.imread(
            osp.normpath('data/cv/binary_blob_wm.png'), cv2.IMREAD_GRAYSCALE)
        blobTool = BlobTool(binary_img)
        blobTool.union_by_orientation(0.03)
        blobs = blobTool.blobs
        self.assertEqual(5, len(blobs))

    def test_blob_union_by_dist_and_orientation(self):
        binary_img = cv2.imread(
            osp.normpath('data/cv/binary_blob_wm.png'), cv2.IMREAD_GRAYSCALE)
        blobTool = BlobTool(binary_img)
        blobTool.union_by_dist_and_orientation(
            dist_thresh=30, orientation_thresh=0.03)
        blobs = blobTool.blobs
        self.assertEqual(6, len(blobs))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestBlobTool("test_blob_filter"),
        TestBlobTool("test_blob_union_by_dist"),
        TestBlobTool('test_blob_union_by_orientation'),
        TestBlobTool("test_blob_union_by_dist_and_orientation"),
        TestBlobTool('test_blob_union')
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
