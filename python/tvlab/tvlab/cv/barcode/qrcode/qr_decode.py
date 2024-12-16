'''
Copyright (C) 2023 TuringVision
'''
import os.path as osp
import numpy as np

__all__ = ['qr_decode', 'CnnQrDecoder']


def qr_decode(img, qr_infos=None, debug=False):
    '''
    In:
        img (h, w)
        qr_infos (list): list of qrcdoe corners and size.
            skip detect qrcode if qr_corners is not None.

            eg [[x1,y1, x2,y2, x3,y3, x4,y4, size], ...]

    Out:
        [{'version': 4,
        'size': 33,
        'score': 98, # 0 ~ 100, qrcdoe quality
        'polygon': [x1,y1,x2,y2,x3,y3,x4,y4],
        'ecc': 'M',
        'ecc_rate': 3, # 0 ~ 100, percent of data use error correction.
        'mask': 0,
        'data_type': 'byte',
        'eci': 0,
        'data': 'hello world',
        'bitmap': np.ndarray((size, size)), # need debug = True
        'err_desc': '',
        },
    '''
    from .impl.cqr_decode import cqr_decode
    if qr_infos is not None:
        qr_infos = np.array(qr_infos).reshape(-1, 9)
        qr_infos = qr_infos.astype(np.int32)
    return cqr_decode(img, qr_infos, debug)


class CnnQrDecoder:
    ''' detect and decode qrcode with cnn binnary model

        Supported QR code version: 3,4,5,6
        Supported QR code box_size: 6~10 pixels

        Usage:
            # init
            d = CnnQrDecoder()

            # open image in gray mode
            img = open_image('path/of/qr.jgp', 'L')

            # roi crop
            img = img[xxx:xxx, xxx:xxx]

            # resize for box_size to 8 pixels
            img = cv2.resize(img, (W, H))

            # run
            out = d.run(img, threshold=0.3)
    '''
    def __init__(self, cnn_binnary_model_path=None, device='cuda'):
        import torch
        from tvdl.segmentation import U2Net
        if cnn_binnary_model_path is None:
            cnn_binnary_model_path = osp.abspath(osp.join(osp.abspath(osp.dirname(osp.realpath(__file__))),
                "./models/cnn_binnary.half.pth"))
        self.model = U2Net(['_'], img_c=1, model_cfg='tiny')
        self.model = self.model.half()
        self.model.load_state_dict(torch.load(cnn_binnary_model_path))
        self.device = device
        if device == 'cpu':
            self.model = self.model.float()
        self.model.freeze()
        self.model = self.model.to(device)

    def binnary(self, img):
        '''
        in:
            img (np.ndarray): HxW (gray) or HxWx3 (rgb), value: 0 ~ 255
            threshold (float): binnary threshold
        out:
            img (np.ndarray) HxW
        '''
        import cv2
        import torch
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        img = img.reshape(1, 1, h, w) / 255.0
        if self.device == 'cpu':
            img = np.float32(img)
        else:
            img = np.float16(img)
        img_t = torch.from_numpy(img).to(self.model.device)
        out = self.model(img_t)[0, 0].cpu().numpy()
        return out

    def run(self, img, threshold=0.5):
        '''
        in:
            img (np.ndarray): HxW (gray) or HxWx3 (rgb), value: 0 ~ 255
            ** When a QR code has 8 pixels in one bit, the effect is usually better **

            threshold (float): binnary threshold
        out:
            [{'version': 4,
            'size': 33,
            'score': 98, # 0 ~ 100, qrcdoe quality
            'polygon': [x1,y1,x2,y2,x3,y3,x4,y4],
            'ecc': 'M',
            'ecc_rate': 3, # 0 ~ 100, percent of data use error correction.
            'mask': 0,
            'data_type': 'byte',
            'eci': 0,
            'data': 'hello world',
            'bitmap': np.ndarray((size, size)), # need debug = True
            'err_desc': '',
            },
        '''
        bin_img = self.binnary(img)
        bin_img = 255 * np.uint8(bin_img > threshold)
        return qr_decode(bin_img)

    def run_all(self, ill, tfms=None, bs=8, num_workers=0, threshold=0.5):
        '''
        in:
            ill: ImageLabelList
            tfms: transform function list, see ImageLabelList.add_tfm
            bs: batch_size
            num_workers:
            threshold (float): binnary threshold
        out:
            [[{'version': 4,
            'size': 33,
            'score': 98, # 0 ~ 100, qrcdoe quality
            'polygon': [x1,y1,x2,y2,x3,y3,x4,y4],
            'ecc': 'M',
            'ecc_rate': 3, # 0 ~ 100, percent of data use error correction.
            'mask': 0,
            'data_type': 'byte',
            'eci': 0,
            'data': 'hello world',
            'bitmap': np.ndarray((size, size)), # need debug = True
            'err_desc': '',
            },..], ...]
        '''
        from fastprogress.fastprogress import progress_bar

        ill.split(1.0)
        ill.set_img_mode('L')
        output = []
        if self.device != 'cpu':
            def _to_half(img):
                import torch
                img = img[None,:,:] / 255.0
                img = np.float16(img)
                img_t = torch.from_numpy(img)
                return img_t

            tfms = tfms + [_to_half]
        _, loader = ill.dataloader(tfms, tfms, bs=bs, num_workers=num_workers)

        for imgs, _ in progress_bar(loader):
            if self.device != 'cpu':
                imgs = imgs.half()
            bin_imgs = self.model(imgs.to(self.device)).cpu().numpy()[:,0]
            for bin_img in bin_imgs:
                bin_img = 255 * np.uint8(bin_img > threshold)
                output.append(qr_decode(bin_img))
        return output

    def detect_threshold(self, img):
        '''
        in:
            img (np.ndarray): HxW (gray) or HxWx3 (rgb), value: 0 ~ 255
        return
            thresholds (list of float): list of good binnary threshold
            infos (list of dict): list of detected qrcode info
        '''
        good_thresholds = []
        good_infos = []
        bin_img_ori = self.binnary(img)
        try_thresholds = [t/100.0 for t in range(5, 100, 5)]
        for threshold in try_thresholds:
            bin_img = 255 * np.uint8(bin_img_ori > threshold)
            info = qr_decode(bin_img)
            if info and 'data' in info[0]:
                good_thresholds.append(threshold)
                good_infos.append(info)
        return good_thresholds, good_infos

    def detect_threshold_all(self, ill, tfms=None, bs=8, num_workers=0, debug=True):
        '''
        in:
            ill: ImageLabelList
            tfms: transform function list, see ImageLabelList.add_tfm
            bs: batch_size
            num_workers:
            debug (bool)
        return
            list of thresholds: list of good binnary thresholds for each image
            list of infos: list of detected qrcode info for each image
        '''
        from fastprogress.fastprogress import progress_bar

        ill.split(1.0)
        ill.set_img_mode('L')
        good_ths_all = []
        good_infos_all = []
        try:
            for img, _ in progress_bar(ill):
                ths, infos = self.detect_threshold(img)
                good_ths_all.append(ths)
                good_infos_all.append(infos)
                if debug and not ths:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.imshow(img)
                    plt.show()
        except KeyboardInterrupt:
            pass
        if debug:
            from tvlab.ui import plot_stack_bar
            all_thresholds = [th for ths in good_ths_all for th in ths]
            bins = list(set(all_thresholds))
            cnts = [all_thresholds.count(d) for d in bins]
            idxs = np.argsort(cnts)[::-1]
            cnts = [cnts[i] for i in idxs]
            bins = [bins[i] for i in idxs]
            plot_stack_bar('', {'threshold': [str(i) for i in bins], 'cnt': cnts}, 'threshold', 'cnt')
        return good_ths_all, good_infos_all
