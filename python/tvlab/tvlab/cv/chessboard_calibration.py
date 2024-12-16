"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring and performing chessboard calibration.
"""

__all__ = ['ChessboardCalibration']

import numpy as np
import cv2
import yaml


class ChessboardCalibration:

    def __init__(self, grid=(22, 16), grid_size=2.5, window_size=(11, 11), eps=0.001):
        """
        Initial the parameters of chessboard images

        Args:
        grid: tuple (width, height), the chessboard inner grid corner size
        grid_size: one grid size(width equals to height) in chessboard, in millimeter(mm)
        window_size: the size of window for calculating corners
        eps: the smaller the eps is, the more accuarate the corner is calculated
        """
        self.eps = eps
        self.window_size = window_size
        self.parameters = {}
        self.objpoints = []
        self.imgpoints = []
        self.grid = grid
        self.grid_size = grid_size
        self.objp = np.zeros((self.grid[0] * self.grid[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.grid[0],
                                    0:self.grid[1]].T.reshape(-1, 2)
        self.scale_ratio = -1
        self.M = None

    def add(self, chess_images):
        """
        add chess_images to do the calibration
        chess_images can be gray_level or RGB color image or image lists

        Args:
        chess_images(ndarray or list): chess image[ndarray] or list of images [ndarray0, ndarray1, ...]

        Returns:
        -1 calibration fail or scale_ratio(float) calibration success
        """
        self.criteria = (cv2.TERM_CRITERIA_EPS
                         + cv2.TERM_CRITERIA_MAX_ITER, 30, self.eps)
        img_lists = []
        if isinstance(chess_images, list):
            for image in chess_images:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                img_lists.append(gray)
        else:
            if len(chess_images.shape) == 3:
                gray = cv2.cvtColor(chess_images, cv2.COLOR_RGB2GRAY)
            else:
                gray = chess_images
            img_lists.append(gray)

        for gray in img_lists:
            ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
            self.h, self.w = gray.shape
            if ret is True:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, self.window_size, (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

        if len(self.objpoints) < 1:
            print("no image found, please add image")
            return -1
        else:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, (self.w, self.h), None, None)
            self.parameters['mtx'] = mtx
            self.parameters['dist'] = dist
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (self.w, self.h), 1, (self.w, self.h))
            self.parameters['newcameramtx'] = newcameramtx
            self.parameters['roi'] = roi

            # select the best imgpoint to do the perspect transform
            best_rec = 0
            best_k = float('inf')
            for i in range(len(self.imgpoints)):
                f_corners = self.imgpoints[i]
                k = abs(f_corners[0][0][1] - f_corners[self.grid[0] - 1][0][1]) / \
                    abs(f_corners[0][0][0] - f_corners[self.grid[0] - 1][0][0])
                if k == 0:
                    best_rec = i
                    break
                if k < best_k:
                    best_k = k
                    best_rec = i

            # do perspective transform
            f_corners = self.imgpoints[best_rec]
            f_corners = np.array(f_corners)
            f_corners = f_corners.squeeze()
            f_corners = cv2.undistortPoints(
                f_corners, mtx, dist, None, newcameramtx)
            src_pts = []
            for i in [0, self.grid[0] - 1, -self.grid[0], -1]:
                src_pts.append(f_corners[i][0])
            w = abs(f_corners[-1][0][0] - f_corners[0][0][0])
            h = abs(f_corners[-1][0][1] - f_corners[0][0][1])

            if w // (self.grid[0] - 1) > h // (self.grid[1] - 1):
                per_grid = h // (self.grid[1] - 1)
            else:
                per_grid = w // (self.grid[0] - 1)

            self.scale_ratio = self.grid_size / per_grid

            src_pts = np.array(src_pts, dtype="float32")
            dst_pts = []
            dst_pts.append([f_corners[0][0][0], f_corners[0][0][1]])
            pts2 = [f_corners[0][0][0] + per_grid * (self.grid[0] - 1), f_corners[0][0][1]]
            dst_pts.append(pts2)
            pts3 = [f_corners[0][0][0], f_corners[0][0][1] + per_grid * (self.grid[1] - 1)]
            dst_pts.append(pts3)
            pts4 = [f_corners[0][0][0] + per_grid * (self.grid[0] - 1), f_corners[0][0][1] + per_grid * (self.grid[1] - 1)]
            dst_pts.append(pts4)
            dst_pts = np.array(dst_pts, dtype="float32")
            self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.pers_ROI = dst_pts

            return self.scale_ratio

    def save(self, save_path):
        with open(save_path, 'wt', encoding='utf-8') as fp:
            config = {'parameters': self.parameters, 'objpoints': self.objpoints,
                      'imgpoints': self.imgpoints, 'grid': self.grid,
                      'criteria': self.criteria, 'objp': self.objp,
                      'h': self.h, 'w': self.w,
                      'grid_size': self.grid_size, 'pers_ROI': self.pers_ROI,
                      'scale_ratio': self.scale_ratio, 'M': self.M}
            yaml.dump(config, fp)

    def load(self, load_path):
        with open(load_path, 'rt', encoding='utf-8') as fp:
            config = yaml.load(fp, Loader=yaml.UnsafeLoader)
            self.parameters = config['parameters']
            self.objpoints = config['objpoints']
            self.imgpoints = config['imgpoints']
            self.grid = config['grid']
            self.criteria = config['criteria']
            self.objp = config['objp']
            self.w = config['w']
            self.h = config['h']
            self.grid_size = config['grid_size']
            self.scale_ratio = config['scale_ratio']
            self.M = config['M']

    def get_scale_ratio(self):
        """
        Return:
        -1 (fail) or one pixel equals to how many mm (successs)
        """
        if len(self.parameters) == 0:
            print("please add before undistort")
            return -1
        return self.scale_ratio

    def undistort(self, img, crop=False, debug_chess=False):
        """
        input a gray level or RGB color image, return an undistort gray level or RGB color image

        Args:
        img(ndarray): an image(gray level or RGB color)
        crop: cropped the undistort image
        debug_chess: show the inner corners of chess image, give 2 more images(ndarray).

        Returns:
        when debug_chess is False
            if successed:
                return: undistort_image
            if failed:
                return: None
        when debug_chess is True
            can find corner: undistort image, chess image with corners, undistort img corners
            can't find cornes: undistort image, None, None
        """
        if len(self.parameters) == 0:
            print("please add before undistort")
            return None

        undis_img = cv2.undistort(
            img, self.parameters['mtx'], self.parameters['dist'], None, self.parameters['newcameramtx'])
        undis_per_img = cv2.warpPerspective(
            undis_img, self.M, (self.w, self.h))
        if crop:
            x, y, w, h = self.parameters['roi']
            undis_per_img = undis_per_img[y:y + h, x:x + w]

        if debug_chess:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            chess_image_dis = img.copy()
            ret, corners = cv2.findChessboardCorners(gray, self.grid, None)
            if ret is False:
                print('cant find inner corners')
                return undis_per_img, None, None

            corners = cv2.cornerSubPix(
                gray, corners, self.window_size, (-1, -1), self.criteria)
            cv2.drawChessboardCorners(chess_image_dis, self.grid, corners, ret)

            corners = cv2.undistortPoints(
                corners, self.parameters['mtx'], self.parameters['dist'], None, self.parameters['newcameramtx'])
            cv2.drawChessboardCorners(undis_img, self.grid, corners, ret)
            return undis_per_img, chess_image_dis, undis_img

        return undis_per_img
