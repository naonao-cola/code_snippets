import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from .basic_detector import BasicDefectDetector


def img_to_tensor(x):
    import torch
    if isinstance(x, np.ndarray):
        if len(x.shape) == 3:
            x = np.transpose(x, (2, 0, 1))
        else:
            x = np.transpose(x, (0, 3, 1, 2))
        x = torch.from_numpy(x.astype(np.float32, copy=False))
    else:
        # N, H, W, C -> N, C, H, W
        x = x.transpose(3, 2).transpose(2, 1).type(torch.float32)
    x.div_(255.0)
    return x


def _default_collate_fn(x):
    """lambda x:list(zip(*x))"""
    return list(zip(*x))


class CnnFeatureBasedDefectDetector(BasicDefectDetector):
    def __init__(self, tile_shape=None, overlap=0.5,
                 border_coef=1.618, device='cuda'):
        '''
        tile_shape (tuple): (h, w)
        overlap (float or int or (h, w)):
        '''
        BasicDefectDetector.__init__(self, device != 'cpu')
        self.device = device
        self._border_coef = border_coef
        self.basemodel = None
        self.fe_model = None
        self.fv_rf = None
        self.fv_border = None
        self.fv_mean = None
        self.fv_std = None
        self.tile_shape = tile_shape
        self.tile_step = None
        if tile_shape:
            tile_h, tile_w = tile_shape

            if isinstance(overlap, tuple):
                overlap_h, overlap_w = overlap
            else:
                overlap_h, overlap_w = overlap, overlap
            if isinstance(overlap_h, float):
                overlap_h = int(tile_h * overlap_h)
            if isinstance(overlap_w, float):
                overlap_w = int(tile_w * overlap_w)

            assert (overlap_h < tile_h and overlap_w < tile_w)
            self.tile_step = (tile_h - overlap_h, tile_w - overlap_w)

    def build_feature_model(self, basemodel='densenet161', fv_rf_s=[16]):
        '''
        In:
            basemodel (str): basemodel name
            fv_rf_s (list): list of output feature level
        Out:
            feature_extractor (nn.Module)
        '''
        from .cnn_feature_extractor import build_feature_model
        return build_feature_model(basemodel=basemodel, fv_rf_s=fv_rf_s)

    def _status_cb(self, desc, percent, cb):
        if cb is not None:
            cb({'desc': desc, 'percent': percent})

    def img_to_tensor(self, x):
        return img_to_tensor(x)

    def show_fv_hist(self, fv, prefix, seed=8888, max_cnt=100000):
        np.random.seed(seed)
        fig, axes = plt.subplots(ncols=8, nrows=1, figsize=(16, 3))
        sample_fv = fv
        if sample_fv.shape[0] >= max_cnt:
            sample_fv = sample_fv[np.random.randint(0, sample_fv.shape[0]-1, max_cnt)]

        sample_ch = np.random.choice(sample_fv.shape[1], 8, replace=False)
        for i, ax in enumerate(axes):
            fv_i = sample_ch[i]
            _ = ax.hist(sample_fv[:, i].cpu().numpy(), bins=100)
            ax.set_title(prefix + f' FV: {fv_i}')
        plt.show()

    def show_err_dist(self, ok_err, ok_fv, ng_err, ng_fv, fv_rf, prefix,
                      max_cnt=1000*1000, debug_ok_pct=99.9):
        import torch
        fig, axes = plt.subplots(ncols=2, figsize=(16, 4))
        sample_err = ok_err
        fv_complexity = torch.sum(ok_fv, dim=1)
        if sample_err.shape[0] >= max_cnt:
            sample_idxs = np.random.randint(0, sample_err.shape[0]-1, max_cnt)
            sample_err = sample_err[sample_idxs]
            fv_complexity = fv_complexity[sample_idxs]

        axes[0].hist(sample_err.flatten().cpu().numpy(), color='b', alpha=0.5, bins=100)
        axes[1].scatter(fv_complexity, sample_err.flatten(), color='b', alpha=0.5, s=1)

        ok_threshold = np.percentile(sample_err.cpu().numpy(), debug_ok_pct)
        axes[0].axvline(ok_threshold, color='r')
        axes[1].axhline(ok_threshold, color='r')


        if ng_err is not None:
            sample_err = ng_err
            fv_complexity = torch.sum(ng_fv, dim=1)
            if sample_err.shape[0] >= max_cnt:
                sample_idxs = np.random.randint(0, sample_err.shape[0]-1, max_cnt)
                sample_err = sample_err[sample_idxs]
                fv_complexity = fv_complexity[sample_idxs]

            axes[0].hist(sample_err.flatten().cpu().numpy(), color='r', alpha=0.5, bins=100)
            axes[1].scatter(fv_complexity, sample_err.flatten(), c='r', alpha=0.5, s=1)

        axes[0].set_title(prefix)
        axes[1].set_title(prefix + ' vs fv_complexity')
        plt.show()

    def _get_ok_ng_idxs(self, fv_shape, label_b, fv_rf, border_h, border_w, debug=False):
        n, c, h, w = fv_shape
        mask = np.zeros((n, h, w), dtype=np.uint8)
        # border 255, ng: 128, good: 0
        # get ng mask
        for j, label in enumerate(label_b):
            if label is not None and isinstance(label, dict) and 'bboxes' in label:
                for box in label['bboxes']:
                    (l,t,r,b) = [int(k/fv_rf) for k in box[:4]]
                    mask[j, t:b, l:r] = 128

        # get border mask
        if border_h > 0:
            mask[:, :border_h] = 255
            mask[:, -border_h:] = 255
        if border_w > 0:
            mask[:, :, :border_w] = 255
            mask[:, :, -border_w:] = 255

        if debug:
            cnt = min(8, n)
            fig, axes = plt.subplots(ncols=cnt, figsize=(2*cnt, 4))
            if cnt == 1:
                axes = [axes]
            for j in range(cnt):
                axes[j].imshow(mask[j], vmin=0, vmax=255, cmap='gray')
                axes[j].set_title(f'mask {j}')
            plt.show()

        mask = mask.flatten()
        # get ng idxs
        ng_idxs = np.where(mask == 128)[0]
        ok_idxs = np.where(mask == 0)[0]
        return ok_idxs, ng_idxs

    def get_ori_fv(self, ibll, tfms, bs=8, workers=8,
                   basemodel='densenet161', fv_rf=16,
                   border=0, percent_cb=None, debug=True):
        '''
        In:

        Out:
        '''
        import torch
        from torch.utils.data import DataLoader
        from fastprogress.fastprogress import progress_bar

        if isinstance(border, (tuple, list)):
            border_h, border_w = border
        else:
            border_h, border_w = border, border

        self.fv_border = (border_h, border_w)

        if self.basemodel is None:
            self.basemodel = basemodel
        else:
            assert self.basemodel == basemodel

        self.fv_rf = fv_rf

        # 0. build fe model
        if self.fe_model is None:
            fe_model = self.build_feature_model(basemodel=basemodel, fv_rf_s=[fv_rf])
            fe_model = fe_model.to(self.device).eval()
            self.fe_model = fe_model

        ibll = ibll.copy()
        ibll.set_tfms(tfms+[img_to_tensor])

        loader = DataLoader(ibll, batch_size=bs, num_workers=workers,
                            collate_fn=_default_collate_fn,
                            shuffle=False, drop_last=False)

        torch.cuda.empty_cache()
        print('========================')
        print('extract_feature...')
        print('========================')
        # 1. get all fv
        ori_fv = None

        # ok idxs
        all_ok_idxs = list()
        # ng idxs
        all_ng_idxs = list()
        with torch.no_grad():
            i = 0
            for img_b, label_b in progress_bar(loader):
                img_b = torch.stack(img_b, 0)
                x_b = img_b.to(self.device)
                if self.tile_shape is not None:
                    x_b, bh, bw, tile_y_b = self.get_tiled_xy(x_b, label_b)
                    label_b = tile_y_b
                ori_fv_s = self.fe_model(x_b)
                fv = ori_fv_s[fv_rf].cpu()
                n, c, h, w = fv.shape
                ok_idxs, ng_idxs = self._get_ok_ng_idxs(fv.shape, label_b, fv_rf,
                                                        border_h, border_w,
                                                        debug=ori_fv is None and debug)
                ok_idxs += i
                ng_idxs += i
                all_ok_idxs += ok_idxs.tolist()
                all_ng_idxs += ng_idxs.tolist()

                if ori_fv is None:
                    n_block = int(n / bs)
                    ori_fv = torch.empty((len(ibll) * n_block * h * w, c),
                                         dtype=torch.float32, device='cpu')
                    print('batch fv shape:', fv.shape)
                # n,c,h,w -> n,h,w,c
                fv = fv.transpose(1, 2).transpose(2, 3)
                # n,h,w,c - > n*h*w,c
                fv = fv.reshape(-1, fv.shape[-1])
                nhw = fv.shape[0]
                ori_fv[i:i+nhw] = fv
                i += nhw
                self._status_cb('extract_feature', int(100*i/len(ibll)), percent_cb)

        # Tensor (N*C*H, W)
        print('all feature shape:', ori_fv.shape)
        # get ok fv
        ori_ok_fv = ori_fv[all_ok_idxs]
        print('ok feature shape:', ori_ok_fv.shape)
        # get ng fv
        ori_ng_fv = None
        if all_ng_idxs:
            ori_ng_fv = ori_fv[all_ng_idxs]
            print('ng feature shape:', ori_ng_fv.shape)

        if debug:
            self.show_fv_hist(ori_ok_fv, 'Ori OK')

        return {'ok': ori_ok_fv, 'ng': ori_ng_fv}

    def _fv_normalize(self, fv, mean=None, std=None):
        '''
        In:
            fv (Tensor): (N, C)

        Out:
        fv (Tensor): (N, C)
        '''
        import torch
        if mean is not None:
            fv -= mean

        if std is not None:
            fv = fv / std

        fv = torch.tanh(fv)
        return fv

    def get_normalized_fv(self, ibll, tfms, bs=8, workers=8,
                          basemodel='densenet161', fv_rf=16,
                          border=0, percent_cb=None, debug=True):
        '''
        In:

        Out:
        '''
        import torch
        ori_fv = self.get_ori_fv(ibll, tfms, bs=bs, workers=workers,
                                 basemodel=basemodel, fv_rf=fv_rf,
                                 border=border, percent_cb=percent_cb,
                                 debug=debug)
        ori_ok_fv = ori_fv['ok']
        # 2. calculate mean/std of fv
        if self.fv_mean is None:
            fv_mean = ori_ok_fv.mean(dim=0)
            print('fv mean[:8]:', fv_mean[:8])
            fv_std = torch.clamp_min(ori_ok_fv.std(dim=0), 1e-6)
            print('fv std[:8]:', fv_std[:8])

            self.fv_mean = fv_mean.to(self.device)
            self.fv_std = fv_std.to(self.device)
        else:
            fv_mean = self.fv_mean.cpu()
            fv_std = self.fv_std.cpu()

        # 3. get normalized fv
        norm_ok_fv = self._fv_normalize(ori_ok_fv, fv_mean, fv_std)

        ori_ng_fv = ori_fv['ng']
        norm_ng_fv = None
        if ori_ng_fv is not None:
            norm_ng_fv = self._fv_normalize(ori_ng_fv, fv_mean, fv_std)

        if debug:
            self.show_fv_hist(norm_ok_fv, 'Norm OK')
        return {'ok': norm_ok_fv, 'ng': norm_ng_fv}

    @abstractmethod
    def forward_with_flatten_ori_fv(self, ori_fv):
        '''
        In:
            ori_fv (torch.Tensor): N, C
        out:
            err (torch.Tensor): N
        '''
        pass

    def get_tiled_xy(self, x_b, y_b=None):
        from ..detection.bbox_overlaps import bbox_overlaps
        tile_h, tile_w = self.tile_shape
        step_h, step_w = self.tile_step
        x_b = x_b.unfold(2, tile_h, step_h).unfold(3, tile_w, step_w)
        n, c, bh, bw, ch, cw = x_b.shape
        x_b = x_b.reshape(n, c, -1, ch, cw)
        x_b = x_b.transpose(1, 2)
        x_b = x_b.reshape(-1, c, ch, cw)
        if y_b is None:
            return x_b, bh, bw
        tile_y_b = []
        iof_threshold = 0.1
        for gt in y_b:
            for i in range(bh):
                for j in range(bw):
                    ys = i * step_h
                    xs = j * step_w
                    labels, bboxes = [], []
                    if gt is not None and isinstance(gt, dict) and 'bboxes' in gt:
                        tbboxes = [[xs, ys, xs+tile_w, ys+tile_h]]
                        overlaps = bbox_overlaps(np.array(gt['bboxes'], np.double),
                                                 np.array(tbboxes, np.double), 'iof')
                        overlaps = overlaps[:, 0]
                        if overlaps.max() > iof_threshold:
                            for j, iof in enumerate(overlaps.tolist()):
                                if iof > iof_threshold:
                                    labels.append(gt['labels'][j])
                                    (l,t,r,b) = gt['bboxes'][j][:4]
                                    l = max(0, min(l - xs, tile_w))
                                    r = max(0, min(r - xs, tile_w))
                                    t = max(0, min(t - ys, tile_h))
                                    b = max(0, min(b - ys, tile_h))
                                    bboxes.append([l,t,r,b]+gt['bboxes'][j][4:])
                    tile_y_b.append({'labels': labels, 'bboxes': bboxes})
        return x_b, bh, bw, tile_y_b

    def forward(self, x):
        '''
        In:
            x (np.ndarry): (N, H, W, C)
        Out:
            amap (torch.Tensor): (N, h, w) # h = H/fv_rf, h = W/fv_rf

        '''
        import torch

        with torch.no_grad():
            n_img = x.shape[0]
            x_t = self.img_to_tensor(x).to(self.device)

            if self.tile_shape is not None:
                x_t, bh, bw = self.get_tiled_xy(x_t)
            ori_fv_s = self.fe_model(x_t)

            ori_fv = ori_fv_s[self.fv_rf]
            n, c, h, w = ori_fv.shape
            flatten_fv = ori_fv.transpose(1, 2).transpose(2, 3)
            flatten_fv = flatten_fv.reshape(-1, flatten_fv.shape[-1])
            vp_err = self.forward_with_flatten_ori_fv(flatten_fv)
            vp_err = vp_err.reshape(n, h, w)
            if self.tile_shape is not None:
                step_h, step_w = self.tile_step
                vp_err = vp_err.cpu()
                vp_err = vp_err.reshape(n_img, -1, h, w)
                img_h, img_w = x.shape[1:3]
                merge_vp_err = torch.full((n_img, img_h//self.fv_rf, img_w//self.fv_rf), 10000.,
                                          dtype=vp_err.dtype, device=vp_err.device)
                vp_step_h = step_h / self.fv_rf
                vp_step_w = step_w / self.fv_rf
                for i in range(bh):
                    for j in range(bw):
                        vp_i = vp_err[:, i * bw + j]
                        sy = int(i * vp_step_h)
                        sx = int(j * vp_step_w)
                        old_vp_i = merge_vp_err[:, sy:sy+h, sx:sx+w]
                        merge_vp_err[:, sy:sy+h, sx:sx+w] = torch.min(vp_i, old_vp_i)
                vp_err = merge_vp_err

            border_h, border_w = self.fv_border
            if border_h > 0:
                vp_err[:, :border_h] = 0
                vp_err[:, -border_h:] = 0
            if border_w > 0:
                vp_err[:, :, :border_w] = 0
                vp_err[:, :, -border_w:] = 0

            return vp_err

    def amap_normalize(self, amap_batch):
        import torch
        for i, amap in enumerate(amap_batch):
            max_value = max(float(amap.max() - amap.min()), 0.0001)
            amap = (amap - amap.min())* (1/max_value)
            amap = amap * 255
            amap_batch[i] = amap

        amap_uint8 = amap_batch.type(torch.uint8)
        amap_uint8 = amap_uint8.cpu().numpy()
        return amap_uint8

    def get_primary_anomaly_map(self, img_batch, border_clean=True, min_v=0, **kwargs):
        '''
        In:
            img_batch: (batch_size, height, width, c) np.uint8
                or list of (height, width, c) np.uint8
            border_clean (bool): do border clean
            min_v (float): clamp min value
        Out:
            amap_batch: (batch_size, height, width) torch.float32
        '''
        if isinstance(img_batch, list):
            img_batch = np.array(img_batch)
        amap_batch = self.forward(img_batch)
        amap_batch = amap_batch.clamp_min(min_v)
        if border_clean:
            amap_batch = self.amap_border_clean(amap_batch)
        return amap_batch

    def amap_border_clean(self, amap_batch):
        if self._border_coef > 1.0:
            border_ratio = 0.05
            bw = int(amap_batch.shape[-1] * border_ratio)
            bh = int(amap_batch.shape[-2] * border_ratio)
            amap_batch[:, :bh] /= np.sqrt(self._border_coef)
            amap_batch[:, :2*bh] /= np.sqrt(self._border_coef)
            amap_batch[:, -bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, -2*bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, :bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, :2*bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -bw:] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -2*bw:] /= np.sqrt(self._border_coef)
        return amap_batch

    def get_anomaly_map_from_rgb(self, img_batch, **kwargs):
        amap_batch = self.get_primary_anomaly_map(img_batch, **kwargs)
        amap_uint8 = self.amap_normalize(amap_batch)
        return amap_uint8

    @classmethod
    def get_bboxes_from_single_amap(cls, amap, min_size=32, origin_shape=None, center_ratio=0.1,
                                    use_max=False):
        '''
        In:
            amap: HxW np.uint8
            min_size: min size of boxes
            origin_shape: auto rescale bboxes to origin shape
        Out:
            defect: [(l,t,r,b), ..] (int)
        '''
        if use_max:
            return BasicDefectDetector.get_bboxes_from_single_amap(amap,
                                                                   min_size=min_size,
                                                                   origin_shape=origin_shape,
                                                                   use_max=use_max)

        if origin_shape is None:
            origin_shape = amap.shape
        _, bin_img = cv2.threshold(amap, amap.max()*0.5, 255,
                                   cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cls.get_bboxes_from_single_binary(bin_img, origin_shape, min_size,
                                                 center_ratio=center_ratio)

    def get_bboxes_from_rgb(self, img_batch, min_size=32):
        '''
        In:
            img_batch: (batch_size, height, width, 3) np.uint8
                    or list of (height, width, 3) np.uint8
            min_size: min size of boxes
        Out:
            defect: [[(l,t,r,b), ..], ...] (int)
        '''
        amap_batch = self.get_anomaly_map_from_rgb(img_batch)
        bboxes_batch = list()
        for i, amap in enumerate(amap_batch):
            ori_shape =  img_batch[i].shape
            bboxes_batch.append(
                self.get_bboxes_from_single_amap(amap, min_size=min_size,
                                                 origin_shape=ori_shape))
        return bboxes_batch

    def get_anomaly_map_from_gray(self, img_batch, **kwargs):
        raise NotImplementedError

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
        raise NotImplementedError

    def get_center_xy_from_gray(self, img_batch):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                    or list of (height, width) np.uint8
        Out:
            defect: [[(x,y), ..], ...] (int)
        '''
        raise NotImplementedError

    @abstractmethod
    def load_from_model_info(self, model_info):
        pass

    def load(self, fname):
        ''' load model from fname
        '''
        import torch
        model_info = torch.load(fname)
        self.basemodel = model_info['basemodel']
        self.fv_rf = model_info['fv_rf_s'][0]

        fe_model = self.build_feature_model(basemodel=self.basemodel, fv_rf_s=[self.fv_rf])
        fe_model = fe_model.to(self.device).eval()
        self.fe_model = fe_model

        self.fv_border = (0, 0)
        fv_border_s = model_info.get('fv_border_s', None)
        if fv_border_s is not None:
            self.fv_border = fv_border_s[self.fv_rf]

        self.fv_mean = None
        fv_mean_s = model_info.get('fv_mean_s', None)
        if fv_mean_s is not None:
            self.fv_mean = fv_mean_s[self.fv_rf].to(self.device)

        self.fv_std = None
        fv_std_s = model_info.get('fv_std_s', None)
        if fv_std_s is not None:
            self.fv_std = fv_std_s[self.fv_rf].to(self.device)

        self.tile_shape = model_info.get('tile_shape', None)
        self.tile_step = model_info.get('tile_step', None)
        self.load_from_model_info(model_info)

    @abstractmethod
    def save_to_model_info(self, model_info):
        pass

    def save(self, fname):
        ''' save model to fname
        '''
        import os
        import os.path as osp
        import torch
        os.makedirs(osp.dirname(fname), exist_ok=True)
        fv_mean_s = None
        if self.fv_mean is not None:
            fv_mean_s = {self.fv_rf: self.fv_mean.cpu()}
        fv_std_s = None
        if self.fv_std is not None:
            fv_std_s = {self.fv_rf: self.fv_std.cpu()}
        model_info = {'basemodel': self.basemodel,
                      'fv_rf_s': [self.fv_rf],
                      'fv_border_s': {self.fv_rf: self.fv_border},
                      'fv_mean_s': fv_mean_s,
                      'fv_std_s': fv_std_s,
                      'tile_shape': self.tile_shape,
                      'tile_step': self.tile_step}
        self.save_to_model_info(model_info)
        torch.save(model_info, fname)
