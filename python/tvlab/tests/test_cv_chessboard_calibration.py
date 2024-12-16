from tvlab import ChessboardCalibration
import unittest
import os
import cv2
import shutil
import os.path as osp
import time


class TestChessboardCalibration(unittest.TestCase):
    root_dir = osp.normpath('./data/cv/distort_img')
    yaml_save_dir = osp.normpath('./models/cv/ccb')
    os.makedirs(yaml_save_dir, exist_ok=True)
    yaml_dir = os.path.join(yaml_save_dir, 'ccb.yaml')
    yaml_dir1 = os.path.join(yaml_save_dir, 'ccb_calib.yaml')
    shutil.rmtree(yaml_dir, ignore_errors=True)
    shutil.rmtree(yaml_dir1, ignore_errors=True)

    def test_init(self):
        ccb = ChessboardCalibration()
        self.assertEqual(ccb.grid, (22, 16))
        self.assertEqual(ccb.grid_size, 2.5)
        self.assertEqual(ccb.window_size, (11, 11))
        self.assertEqual(ccb.eps, 0.001)
        self.assertIsInstance(ccb.parameters, dict)
        self.assertIsInstance(ccb.objpoints, list)
        self.assertIsInstance(ccb.imgpoints, list)

        ccb = ChessboardCalibration((5, 5))
        self.assertEqual(ccb.grid, (5, 5))

    def test_add(self):
        ccb = ChessboardCalibration(grid=(47, 34))

        images_list = []
        gray_images_list = []
        image_files_path = []
        for image_file_path in os.listdir(self.root_dir):
            if '.png' in image_file_path:
                image_files_path.append(os.path.join(self.root_dir, image_file_path))
                img = cv2.imread(os.path.join(self.root_dir, image_file_path))
                img = img[:, :, :: -1]
                images_list.append(img)
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                gray_images_list.append(gray_img)

        ret = ccb.add(images_list)
        self.assertEqual(len(ccb.imgpoints), 8)
        self.assertEqual(len(ccb.objpoints), 8)
        self.assertNotEqual(ret, -1)
        self.assertTrue('mtx' in ccb.parameters)
        self.assertTrue('dist' in ccb.parameters)
        self.assertTrue('newcameramtx' in ccb.parameters)
        self.assertTrue('roi' in ccb.parameters)
        self.assertTrue(ccb.M is not None)
        ccb.save(self.yaml_dir1)

        ret = ccb.add(gray_images_list)
        self.assertEqual(len(ccb.imgpoints), 16)
        self.assertEqual(len(ccb.objpoints), 16)
        self.assertNotEqual(ret, -1)
        self.assertTrue('mtx' in ccb.parameters)
        self.assertTrue('dist' in ccb.parameters)
        self.assertTrue('newcameramtx' in ccb.parameters)
        self.assertTrue('roi' in ccb.parameters)
        self.assertTrue(ccb.M is not None)

        for image_file_path in image_files_path:
            img = cv2.imread(image_file_path)
            img = img[:, :, :: -1]
            ccb.add(img)
        self.assertEqual(len(ccb.imgpoints), 24)
        self.assertEqual(len(ccb.objpoints), 24)
        self.assertNotEqual(ret, -1)
        self.assertTrue('mtx' in ccb.parameters)
        self.assertTrue('dist' in ccb.parameters)
        self.assertTrue('newcameramtx' in ccb.parameters)
        self.assertTrue('roi' in ccb.parameters)
        self.assertTrue(ccb.M is not None)

        for image_file_path in image_files_path:
            img = cv2.imread(image_file_path, 0)
            ccb.add(img)
        self.assertEqual(len(ccb.imgpoints), 32)
        self.assertEqual(len(ccb.objpoints), 32)
        self.assertNotEqual(ret, -1)
        self.assertTrue('mtx' in ccb.parameters)
        self.assertTrue('dist' in ccb.parameters)
        self.assertTrue('newcameramtx' in ccb.parameters)
        self.assertTrue('roi' in ccb.parameters)
        self.assertTrue(ccb.M is not None)

    def test_save(self):
        ccb = ChessboardCalibration(grid=(47, 34))
        for image_file_path in os.listdir(self.root_dir):
            if '.png' in image_file_path:
                img = cv2.imread(os.path.join(self.root_dir, image_file_path))
                ccb.add(img)
        ccb.save(self.yaml_dir)
        self.assertTrue(os.path.exists(self.yaml_dir))

    def test_load(self):
        ccb = ChessboardCalibration(grid=(47, 34))
        ccb.load(self.yaml_dir)
        self.assertEqual(len(ccb.imgpoints), 8)
        self.assertEqual(len(ccb.objpoints), 8)
        self.assertNotEqual(len(ccb.parameters), 0)

    def test_get_scale_ratio(self):
        ccb = ChessboardCalibration(grid=(47, 34))
        scale = ccb.get_scale_ratio()
        self.assertEqual(scale, -1)

        ccb.load(self.yaml_dir1)
        scale = ccb.get_scale_ratio()
        scale_flag = True
        if scale - 0.04629 > 0.0001:
            scale_flag = False
        self.assertTrue(scale_flag)

    def test_undistort(self):
        ccb = ChessboardCalibration(grid=(47, 34))
        img_path1 = os.path.join(self.root_dir, 'chess0.png')
        img = cv2.imread(img_path1)
        img = img[:, :, :: -1]
        un_img = ccb.undistort(img)
        self.assertTrue(un_img is None)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        un_img = ccb.undistort(img_gray)
        self.assertTrue(un_img is None)

        all_time = 0
        ccb.load(self.yaml_dir1)
        for image_file_path in os.listdir(self.root_dir):
            if '.png' in image_file_path:
                img = cv2.imread(os.path.join(self.root_dir, image_file_path))
                img = img[:, :, :: -1]
                start = time.time()
                und_img = ccb.undistort(img)
                all_time += time.time() - start
                self.assertEqual(und_img.shape[:2], (ccb.h, ccb.w))
                und_img = ccb.undistort(img, crop=True)
                roi = ccb.parameters['roi']
                self.assertEqual(und_img.shape[:2], (roi[3], roi[2]))

        avg_time = all_time / len(ccb.objpoints)
        time_flag = True
        if avg_time > 0.2:
            time_flag = False
        self.assertTrue(time_flag)

        img0 = cv2.imread(img_path1)
        img0 = img0[:, :, :: -1]
        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img0, debug_chess=True)
        self.assertEqual(len(und_img.shape), 3)
        self.assertTrue(und_img.shape[:2], (ccb.h, ccb.w))
        self.assertTrue(dis_img_cor is None)
        self.assertTrue(un_img_cor is None)

        img_gray = cv2.imread(img_path1, 0)
        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img_gray, debug_chess=True)
        self.assertEqual(len(und_img.shape), 2)
        self.assertTrue(und_img.shape[:2], (ccb.h, ccb.w))
        self.assertTrue(dis_img_cor is None)
        self.assertTrue(un_img_cor is None)

        img_path2 = os.path.join(self.root_dir, 'chess10.png')
        img1 = cv2.imread(img_path2)
        img1 = img1[:, :, ::-1]
        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img1, debug_chess=True)
        self.assertEqual(len(und_img.shape), 3)
        self.assertEqual(und_img.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(dis_img_cor.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(un_img_cor.shape[:2], (ccb.h, ccb.w))

        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img1, crop=True, debug_chess=True)
        self.assertEqual(len(und_img.shape), 3)
        self.assertEqual(und_img.shape[:2], (roi[3], roi[2]))
        self.assertEqual(dis_img_cor.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(un_img_cor.shape[:2], (ccb.h, ccb.w))

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img1_gray, debug_chess=True)
        self.assertEqual(len(und_img.shape), 2)
        self.assertEqual(und_img.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(dis_img_cor.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(un_img_cor.shape[:2], (ccb.h, ccb.w))

        und_img, dis_img_cor, un_img_cor = ccb.undistort(
            img1_gray, crop=True, debug_chess=True)
        self.assertEqual(len(und_img.shape), 2)
        self.assertEqual(und_img.shape[:2], (roi[3], roi[2]))
        self.assertEqual(dis_img_cor.shape[:2], (ccb.h, ccb.w))
        self.assertEqual(un_img_cor.shape[:2], (ccb.h, ccb.w))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestChessboardCalibration("test_init"),
        TestChessboardCalibration("test_add"),
        TestChessboardCalibration("test_save"),
        TestChessboardCalibration("test_load"),
        TestChessboardCalibration("test_get_scale_ratio"),
        TestChessboardCalibration("test_undistort")
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
