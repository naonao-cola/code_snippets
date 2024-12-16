'''
Copyright (C) 2023 TuringVision

Convolutional category model visualizations.
What is the focus of the model on the image?
'''

from .guided_backprop import GuidedBackprop

__all__ = ['CategoryModelVis']

class CategoryModelVis:
    '''
    Convolutional category model visualizations
    '''
    def __init__(self, learner):
        '''
        learner: fastai Learner
        '''
        self._learner = learner

    def get_heatmap(self, idxs, with_src=False):
        ''' get heatmap for the input images
        In:
            idxs: (list) index in valid dataset
            with_src: (bool) return source image when it's True
        Out:
            heatmap_list: (list) list of heatmap
        '''
        with GuidedBackprop(self._learner.model) as gbp:
            ds = self._learner.data.valid_ds
            src_img_list = []
            heatmap_list = []
            for i in idxs:
                img, label = ds[i]
                src_img = img.data
                if with_src:
                    np_img = src_img.cpu().numpy()
                    np_img = np_img.transpose(1, 2, 0)
                    src_img_list.append(np_img)
                heatmap = gbp.get_heatmap(src_img, label.data)
                heatmap_list.append(heatmap)
        if with_src:
            return heatmap_list, src_img_list
        return heatmap_list

    def get_heatmap_on_img(self, idxs, alpha=0.5):
        ''' get heatmap ont img for the input images
        In:
            idxs: (list) index in valid dataset
            alpha: (float) merge ratio for heatmap and src_img
        Out:
            heatmap_on_img_list: (list) list of merge image
        '''
        heatmap_list, src_img_list = self.get_heatmap(idxs, with_src=True)
        import cv2
        from PIL import Image
        import numpy as np
        import matplotlib.cm as mpl_color_map

        heatmap_on_img_lsit = []
        for heatmap, src_img in zip(heatmap_list, src_img_list):
            heatmap = cv2.blur(heatmap, (7, 7))
            color_map = mpl_color_map.get_cmap('magma')
            heatmap = color_map(heatmap)
            heatmap_on_img = src_img * (1 - alpha) + heatmap[:, :, :3] * alpha
            heatmap_on_img = heatmap_on_img * 255
            heatmap_on_img = heatmap_on_img.astype(np.uint8)
            heatmap_on_img_lsit.append(heatmap_on_img)
        return heatmap_on_img_lsit

    def show(self, ill, idxs=None, desc_list=None, alpha=0.5, **kwargs):
        ''' show dataset with heatmap
        In:
            ill: ImageLabelList
            idxs: (list) index in valid dataset
            desc_list: (list) list of str
            alpha: (float) merge ratio for heatmap and src_img
        Out:
            ImageCleaner
        '''
        from .image_data import ImageLabelList
        from ..ui import ImageCleaner
        if idxs is None:
            idxs = list(range(len(ill)))

        model_vis = self
        class _HeatmapLabelList(ImageLabelList):
            def __getitem__(self, idx):
                heatmap_on_img_list = model_vis.get_heatmap_on_img([idx], alpha)
                return heatmap_on_img_list[0], self.y[idx]

        new_ill = ill.copy()
        new_ill.__class__ = _HeatmapLabelList
        return ImageCleaner(new_ill, find_idxs=idxs, desc_list=desc_list, **kwargs)
