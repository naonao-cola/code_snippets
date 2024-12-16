'''
Copyright (C) 2023 TuringVision

basic defect detector
'''
import cv2
from enum import IntEnum
from tqdm.auto import tqdm
import numpy as np
from abc import ABC, abstractmethod

__all__ = ['rgb2gray_gpu', 'rgb2gray_cpu', 'BasicDefectDetector']


def rgb2gray_cpu(color_image):
    '''
    In: HxWx3 np.uint8 (0~255)
    Out: HxW np.float32
    '''
    MIN_CONTRAST = 0.5
    image = np.empty(color_image.shape[:3], np.float32)
    for c in range(3):
        c_min = np.percentile(color_image[..., c], 1)
        c_max = np.percentile(color_image[..., c], 99)
        c_max = max(c_min+MIN_CONTRAST, c_max)
        c_avg = max(np.mean(color_image[..., c]), c_min + 1e-5)
        c_avg2 = (c_avg-c_min)/(c_max-c_min)
        power = np.log(0.5) / np.log(c_avg2)
        power = np.clip(power, 0.5, 2.0)
        channel = np.clip((color_image[..., c]-c_min)/(c_max-c_min), 0, 1)
        image[..., c] = channel**power
    image = np.mean(image, axis=2)
    image = 255*(image - image.min() / (image.max() - image.min()))
    image = image.astype(np.uint8)
    return image

def _hist_percentile(percent_idx, bin_edges, percent):
    for i, cur_percent in enumerate(percent_idx):
        if cur_percent >= percent:
            return bin_edges[i]
    return bin_edges[-1]


def rgb2gray_gpu(color_image):
    '''
    In: HxWx3 np.uint8 (0~255)
    Out: HxW
    '''
    import torch
    MIN_CONTRAST = 0.5
    MAX_VALUE = 255.0
    HIST_BINS = 256
    image = torch.empty(color_image.shape[:3], dtype=torch.float32, device='cuda')
    color_image_gpu = torch.from_numpy(color_image).cuda()
    color_image_gpu = color_image_gpu.type(torch.float32)
    sub_image = color_image_gpu[::2, ::2]
    for c in range(3):
        hist = torch.histc(sub_image[:, :, c], bins=HIST_BINS, min=0.0, max=MAX_VALUE)
        hist = hist.cpu().numpy()
        bin_edges = np.linspace(0.0, MAX_VALUE, HIST_BINS, dtype=np.float32)
        total_sum = sub_image.shape[0] * sub_image.shape[1]
        percent_idx = hist.cumsum() * 100 / total_sum
        c_min = _hist_percentile(percent_idx, bin_edges, 1)
        c_max = _hist_percentile(percent_idx, bin_edges, 99)
        mean = (hist * bin_edges).sum() / total_sum
        c_max = max(c_min+MIN_CONTRAST, c_max)
        c_avg = max(mean, c_min + 1e-5)
        c_avg2 = (c_avg-c_min)/(c_max-c_min)
        power = np.log(0.5) / np.log(float(c_avg2))
        power = np.clip(power, 0.5, 2.0)
        channel = (color_image_gpu[..., c]-c_min)/(c_max-c_min)
        channel = torch.clamp(channel, min=0.0, max=1.0)
        image[..., c] = torch.pow(channel, power)
    image = torch.mean(image, dim=2)
    image = 255*(image - image.min() / (image.max() - image.min()))
    image = image.type(torch.uint8)
    image = image.cpu().numpy()
    return image


def _bboxes_batch_to_xy_batch(bboxes_batch):
    xy_batch = [((box[0]+box[2])//2, (box[1]+box[3])//2) if box else None
                for bboxes in bboxes_batch for box in bboxes]
    return xy_batch


def is_bbox_ltwh_overlay(l_ltwh, r_ltwh, min_size=32):
    ll, lt, lw, lh = l_ltwh
    rl, rt, rw, rh = r_ltwh
    if lw < min_size:
        ll = ll + (lw - min_size) // 2
        lw = min_size
    lr = ll + lw
    if lh < min_size:
        lt = lt + (lh - min_size) // 2
        lh = min_size
    lb = lt + lh
    if rw < min_size:
        rl = rl + (rw - min_size) // 2
        rw = min_size
    rr = rl + rw
    if rh < min_size:
        rt = rt + (rh - min_size) // 2
        rh = min_size
    rb = rt + rh
    if max(ll, rl) < min(lr, rr) and max(lt, rt) < min(lb, rb):
        return True

    return False


def merge_ltwh(ltwh_list, min_size, img_w, img_h):
    l = min([ltwh[0] for ltwh in ltwh_list])
    t = min([ltwh[1] for ltwh in ltwh_list])
    r = max([ltwh[0]+ltwh[2] for ltwh in ltwh_list])
    b = max([ltwh[1]+ltwh[3] for ltwh in ltwh_list])
    if (r - l) < min_size:
        l = (r + l - min_size) // 2
        r = l + min_size
    if (b - t) < min_size:
        t = (t + b - min_size) // 2
        b = t + min_size
    l = max(0, l)
    r = min(img_w-1, r)
    t = max(0, t)
    b = min(img_h-1, b)
    return l, t, r - l, b - t


def ltwh_score(ltwh, img_w, img_h, center_ratio=0.1):
    center_dist = ((ltwh[0] + ltwh[2]//2 - img_w//2)**2 + (ltwh[1] + ltwh[3]//2 - img_h//2)**2)**0.5
    # 0 ~ 1.0
    center_score = 1 - center_dist / ((img_w//2)**2 + (img_h//2)**2)**0.5
    # 0 ~ 1.0
    box_size_score = ltwh[2] * ltwh[3] / (img_w * img_h)

    return center_ratio * center_score + (1-center_ratio) * box_size_score


class BasicDefectDetector(ABC):
    '''
    '''
    def __init__(self, use_gpu=False):
        self._use_gpu = use_gpu

    def rgb_to_gray(self, rgb_batch):
        gray_img_batch = list()
        for img in rgb_batch:
            if self._use_gpu:
                gray_img = rgb2gray_gpu(img)
            else:
                gray_img = rgb2gray_cpu(img)
            gray_img_batch.append(gray_img)
        return gray_img_batch

    def get_anomaly_map_from_rgb(self, img_batch):
        '''
        In:
            img: img_batch: (batch_size, height, width, c) np.uint8
            or list of (height, width, c) np.uint8
        Out:
            amap: (NxHxW np.uint8)
        '''
        gray_img_batch = self.rgb_to_gray(img_batch)
        return self.get_anomaly_map_from_gray(gray_img_batch)

    @abstractmethod
    def get_anomaly_map_from_gray(self, img_batch):
        '''
        In:
            img: numpy img (NxHxW)
        Out:
            amap: (NxHxW np.uint8)
        '''
        pass

    @classmethod
    def get_bboxes_from_single_binary(cls, bin_img, origin_shape=None,
                                      min_size=32, center_ratio=0.1):
        '''
        In:
            amap: HxW np.uint8
            origin_shape:  for scale
            min_size: min size of boxes
        Out:
            defect: [(l,t,r,b), ..] (int)
        '''
        img_h, img_w = origin_shape[:2] if origin_shape else bin_img.shape[:2]
        ret = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = ret[0]
        if len(ret)==3:
            contours = ret[1]
        if not contours:
            return None

        bboxes_result = [cv2.boundingRect(cnt) for cnt in contours]
        bboxes_result = np.array(bboxes_result, dtype=np.float32)
        if origin_shape:
            h_ratio = img_h / bin_img.shape[0]
            w_ratio = img_w / bin_img.shape[1]
            bboxes_result[:, ::2] *= w_ratio
            bboxes_result[:, 1::2] *= h_ratio
        bboxes_result = bboxes_result.astype(np.int32)
        bboxes_result = bboxes_result.tolist()

        # Merge the closer boxes
        while True:
            close_status = np.arange(len(bboxes_result))
            for i, left in enumerate(bboxes_result):
                for j, right in enumerate(bboxes_result[i+1:]):
                    if is_bbox_ltwh_overlay(left, right, min_size):
                        x = i+j+1
                        close_status[i] = close_status[x] = min(close_status[i], close_status[x])

            merged_bboxes_result = list()
            for i in range(max(close_status)+1):
                pos = np.where(close_status == i)[0]
                if pos.size == 0:
                    continue
                ltwh = merge_ltwh([bboxes_result[i] for i in pos], min_size, img_w, img_h)
                merged_bboxes_result.append(ltwh)
            if len(bboxes_result) != len(merged_bboxes_result):
                bboxes_result = merged_bboxes_result
            else:
                break

        # boxes sorted with the proximity of the center and boxes size.
        # TODO: Calculate the sum of anomaly in the box to replace box size
        merged_bboxes_result.sort(key=lambda ltwh: ltwh_score(ltwh, img_w,
                                                              img_h, center_ratio),
                                  reverse=True)
        bboxes_ltrb = list()
        for ltwh in merged_bboxes_result:
            ltrb = [ltwh[0], ltwh[1], ltwh[0]+ltwh[2], ltwh[1]+ltwh[3]]
            bboxes_ltrb.append(ltrb)
        return bboxes_ltrb

    @classmethod
    def get_bboxes_from_single_amap(cls, amap, min_size=32, origin_shape=None,
                                    downscale_factor=12, center_ratio=0.1,
                                    use_max=False):
        '''
        In:
            amap: HxW np.uint8
            min_size: min size of boxes
            downscale_factor: downscale factor
        Out:
            defect: [(l,t,r,b), ..] (int)
        '''
        if use_max:
            img_h, img_w = origin_shape[:2] if origin_shape else amap.shape[:2]
            h_ratio = img_h / amap.shape[0]
            w_ratio = img_w / amap.shape[1]
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(amap)
            _x, _y = maxLoc
            _x *= w_ratio
            _y *= h_ratio
            half_size = min_size//2
            l = _x - half_size
            t = _y - half_size
            r = _x + half_size
            b = _y + half_size
            bboxes = [[l,t,r,b]]
            return bboxes

        if origin_shape is None:
            origin_shape = amap.shape
        img_h, img_w = amap.shape[:2]
        amap_scaled = cv2.resize(amap, (img_w//downscale_factor, img_h//downscale_factor),
                                 interpolation=cv2.INTER_AREA)
        _, bin_img = cv2.threshold(amap_scaled, amap_scaled.max()*0.5, 255,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cls.get_bboxes_from_single_binary(bin_img, origin_shape, min_size,
                                                 center_ratio=center_ratio)

    def get_bboxes_from_gray(self, img_batch, min_size=32, downscale_factor=12):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                    or list of (height, width) np.uint8
            min_size: min size of boxes
            downscale_factor: downscale factor
        Out:
            defect: [[(l,t,r,b), ..], ...] (int)
        '''
        amap_batch = self.get_anomaly_map_from_gray(img_batch)
        bboxes_batch = list()
        for amap in amap_batch:
            bboxes_batch.append(
                self.get_bboxes_from_single_amap(amap,
                                                 min_size=min_size,
                                                 downscale_factor=downscale_factor))
        return bboxes_batch

    def get_bboxes_from_rgb(self, img_batch, min_size=32):
        '''
        In:
            img_batch: (batch_size, height, width, 3) np.uint8
                    or list of (height, width, 3) np.uint8
            min_size: min size of boxes
        Out:
            defect: [[(l,t,r,b), ..], ...] (int)
        '''
        gray_img_batch = self.rgb_to_gray(img_batch)
        return self.get_bboxes_from_gray(gray_img_batch, min_size)

    def get_center_xy_from_gray(self, img_batch):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                    or list of (height, width) np.uint8
        Out:
            defect: [[(x,y), ..], ...] (int)
        '''
        bboxes_batch = self.get_bboxes_from_gray(img_batch)
        return _bboxes_batch_to_xy_batch(bboxes_batch)

    def get_center_xy_from_rgb(self, img_batch):
        '''
        In:
            img_batch: (batch_size, height, width, 3) np.uint8
                    or list of (height, width) np.uint8
        Out:
            defect: [[(x,y), ..], ...] (int)
        '''
        bboxes_batch = self.get_bboxes_from_rgb(img_batch)
        return _bboxes_batch_to_xy_batch(bboxes_batch)
