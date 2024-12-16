from tvlab import *
import unittest
import cv2
import os.path as osp


class TestQrdecode(unittest.TestCase):
    def test_qrdecode(self):
        img = cv2.imread(osp.normpath('./data/002.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        re = qr_decode(img)
        self.assertEqual(re, [
            {'version': 5, 'size': 37, 'score': 100, 'polygon': [141, 49, 893, 49, 893, 801, 141, 801], 'ecc': 'H',
             'ecc_rate': 14, 'mask': 2, 'data_type': 'byte', 'eci': 0,
             'data': 'https://u.wechat.com/MDUkBA1mMpNDdr8_dBdufz0'}])

    def test_cnn_qrdecode(self):
        img = open_image(osp.normpath('./data/002.jpg'), 'L')
        img = cv2.resize(img, (350, 300))
        d = CnnQrDecoder()
        re = d.run(img)
        self.assertEqual(re, [
            {'version': 5, 'size': 37, 'score': 100, 'polygon': [47, 17, 298, 15, 298, 274, 48, 272], 'ecc': 'H',
             'ecc_rate': 14, 'mask': 2, 'data_type': 'byte', 'eci': 0,
             'data': 'https://u.wechat.com/MDUkBA1mMpNDdr8_dBdufz0'}])


if __name__ == '__main__':
    unittest.main()
