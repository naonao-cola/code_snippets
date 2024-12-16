'''
Copyright (C) 2023 TuringVision

Jupyter tool for image with bboxes view and relabeling
'''
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
from ..utils import get_notebook_url
from .image_cleaner import ImageCleaner


__all__ = ['ImageBBoxCleaner']


def _y2source(y, label_level=False):
    from bokeh.models import ColumnDataSource
    if not y or 'labels' not in y or 'bboxes' not in y:
        y = {'labels': list(), 'bboxes': list(), 'colors': list()}
    bboxes = y['bboxes']
    labels = y['labels']
    colors = y.get('colors', ['springgreen'] * len(labels))
    levels = y.get('levels', [5] * len(labels))
    x, y, w, h = list(), list(), list(), list()
    l, t, text = list(), list(), list()
    color = list()
    for box, label, c, level in zip(bboxes, labels, colors, levels):
        l.append(box[0])
        t.append(box[1])
        x.append((box[0] + box[2]) / 2)
        y.append((box[1] + box[3]) / 2)
        w.append(box[2] - box[0])
        h.append(box[3] - box[1])
        if label_level:
            label = label + '-' + str(level)
        text.append(label)
        color.append(c)
    source = ColumnDataSource(dict(x=x, y=y, w=w, h=h, l=l, t=t, text=text, color=color))
    return source


def _source2y(source, label_level=False):
    labels = source.data['text']
    levels = []
    if label_level:
        new_labels = []
        for label in labels:
            label, level = label.split('-')
            new_labels.append(label)
            levels.append(int(level))
        labels = new_labels
    l = source.data['l']
    t = source.data['t']
    w = source.data['w']
    h = source.data['h']
    colors = source.data['color']
    bboxes = [[_l, _t, _l+_w, _t+_h] for _l,_t,_w,_h in zip(l,t,w,h)]

    y = {'labels': labels, 'bboxes': bboxes, 'colors': colors}
    if label_level:
        y['levels'] = levels
    return y


def _check_y_change(y1, y2, label_level=False):
    if y1['labels'] != y2['labels'] or y1['bboxes'] != y2['bboxes']:
        return True
    if label_level:
        if 'levels' not in y1:
            return True
        if 'levels' in y1 and 'levels' in y2 and y1['levels'] != y2['levels']:
            return True
    return False


class ImageBBoxCleaner(ImageCleaner):
    #for speedup bokeh show image
    IMG_CACHE_DIR = 'img_bbox_cleaner_cache'

    '''Graphical image annotation tool for bboxes detection task.
    '''
    def __init__(self, ibll, find_idxs=None,
                 desc_list=None,
                 notebook_url=None,
                 ncols=2, nrows=1, labelset=None,
                 box_tfm=None,
                 max_size=600,
                 xml_path=None,
                 active_zoom=False,
                 share_xy=False,
                 label_level=None,
                 img_format='jpeg',
                 img_quality=95,
                 port=None):
        '''
        # Arguments:
            ibll: (ImageBBoxLabelList)
            find_idxs: (list) index in ill for relabeling
            desc_list: (list) description of each find item
            notebook_url: (str) 'ipaddr:port' for bokeh data sync in notebook
                        or call set_notebook_url('ipaddr:port') first
            ncols: (int) number of column
            nrows: (int) number of row
            labelset: (list) list of label, get from ill when it's None
            box_tfm: transform function for restore polygon before save change.
                Default will rescale the polygon to the original image size if
                box_tfm is None.
                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bpolygon by adding a `box_tfm` function.

            xml_path: (str) xml output path (only save changed image)
            active_zoom (bool): active wheel zoom tool
            share_xy (bool): share x_range/y_range for column x row views
            img_format: one of 'jpeg', 'png', 'bmp'
                    'jpeg' for high speed browse image,
                    'png' and 'bmp' for high resolution browse image.
            port: (int) for jupyter-notebook in docker,
                  The port Numbers of the physical machine and the container must be the same
        '''
        if not notebook_url:
            notebook_url = get_notebook_url()
            if not notebook_url:
                notebook_url = 'localhost:8888'

        self._ibll = ibll
        self._new_ibll = ibll.copy()
        self._relabel_info = None
        self._img_format = img_format
        self._img_quality = img_quality
        self._find_idxs = find_idxs
        self._desc_list = desc_list
        self._notebook_url = notebook_url
        self._bokeh_port = port
        self._img_cache_dir = ImageBBoxCleaner.IMG_CACHE_DIR
        self._labelset = ibll.labelset() if labelset is None else labelset
        self._active_zoom = active_zoom
        self._share_xy = share_xy
        if label_level is None:
            label_level = True if 'levels' in ibll.y[0] else False
        self._label_level = label_level
        if not self._labelset:
            self._labelset = ['object']

        self._box_tfm = box_tfm
        self._ncols = ncols
        self._nrows = nrows
        self._max_size = max_size
        self._xml_path = xml_path
        # for avoid image_url cache
        self._rand_key = str(np.random.randint(0, 1e8))
        self._op_id = 0
        self._index = 0
        if self._find_idxs is None:
            self._find_idxs = list(range(len(ibll)))
        self._source_list = None
        self._old_y_list = None
        self._scale_ratio_list = None
        self._ori_image_shape_list = None
        self._render()

    @property
    def find_idxs(self):
        return self._find_idxs

    def get_ibll(self):
        ''' get modified ibll
        '''
        self._save_relabel_result()
        new_ibll = self._new_ibll.copy()
        new_ibll.y = new_ibll.y.filter_by_label(lambda l: l in self._labelset)
        return new_ibll

    def get_info(self):
        raise NotImplementedError

    def to_json(self, json_path):
        raise NotImplementedError

    def _clean_cache_dir(self):
        import shutil
        shutil.rmtree(self._img_cache_dir, ignore_errors=True)
        os.makedirs(self._img_cache_dir, exist_ok=True)

    def _show_bokeh_bbox_edit_ui(self):
        from bokeh.io import show
        from .bokeh_bbox_edit_tool import get_bokeh_bbox_edit_app
        from ..category.image_data import get_image_res
        self._clean_cache_dir()
        img_path_list = list()
        source_list = list()
        old_y_list = list()
        scale_ratio_list = list()
        ori_image_shape_list = list()
        desc_list = list()
        for n in range(self._nrows):
            for c in range(self._ncols):
                i = n * self._ncols + c + self._index
                if i >= len(self._find_idxs):
                    continue
                index = self._find_idxs[i]
                img, y = self._new_ibll[index]
                img_path = self._new_ibll.x[index]
                img_name = osp.splitext(osp.basename(img_path))[0]
                img_cache_name = str(i)+'_'+str(self._op_id)+'_'+self._rand_key+ img_name + '.' + self._img_format
                self._op_id += 1
                img_cache_name = img_cache_name.replace('#', '_')
                img_cache_path = osp.join(self._img_cache_dir, img_cache_name)
                Image.fromarray(img).save(img_cache_path, quality=self._img_quality)
                # save img to local dir
                img_path_list.append(img_cache_path)
                ori_w, ori_h = get_image_res(img_path)
                cur_h, cur_w = img.shape[:2]
                scale_h = ori_h / cur_h
                scale_w = ori_w / cur_w
                scale_ratio_list.append((scale_h, scale_w))
                ori_image_shape_list.append((ori_h, ori_w))
                old_y_list.append(y)
                source_list.append(_y2source(y, self._label_level))
                desc = self._desc_list[i] if self._desc_list is not None else osp.basename(img_path)
                desc_list.append(desc)

        bk_app = get_bokeh_bbox_edit_app(img_path_list,
                                         source_list=source_list,
                                         labelset=self._labelset,
                                         desc_list=desc_list,
                                         ncols=self._ncols,
                                         max_size=self._max_size,
                                         active_zoom=self._active_zoom,
                                         share_xy=self._share_xy,
                                         label_level=self._label_level)
        self._ori_image_shape_list = ori_image_shape_list
        self._scale_ratio_list = scale_ratio_list
        self._old_y_list = old_y_list
        self._source_list = source_list
        show(bk_app, notebook_url=self._notebook_url, port=self._bokeh_port)

    def _save_relabel_result(self):
        from ..detection.bbox_label import BBoxLabel

        def _scale_box_tfm(box, scale_ratio):
            box = box[:4]
            box = np.array(box, dtype=np.float32).reshape(-1, 2)
            box[:, 0] *= scale_ratio[1]
            box[:, 1] *= scale_ratio[0]
            return box.flatten().tolist()

        if self._source_list:
            change_x, change_y = [], []
            for i, source in enumerate(self._source_list):
                index = self._find_idxs[i+self._index]
                old_y = self._old_y_list[i]
                new_y = _source2y(source, self._label_level)
                old_y_gt = BBoxLabel(old_y).filter_by_label(lambda l: l in self._labelset)
                new_y_gt = BBoxLabel(new_y).filter_by_label(lambda l: l in self._labelset)
                scale_ratio = self._scale_ratio_list[i]
                ori_shape = self._ori_image_shape_list[i]
                if _check_y_change(old_y_gt, new_y_gt, self._label_level):
                    if self._box_tfm:
                        new_y['bboxes'] = [self._box_tfm(box, ori_shape)
                                           for box in new_y['bboxes']]
                    else:
                        new_y['bboxes'] = [_scale_box_tfm(box, scale_ratio)
                                           for box in new_y['bboxes']]
                    self._new_ibll.y[index] = new_y
                    change_x.append(self._new_ibll.x[index])
                    change_y.append(new_y_gt)
            if self._xml_path and change_x:
                _ibll = self._ibll.__class__(change_x, change_y)
                _ibll.to_pascal_voc(self._xml_path)

    def _update_index(self, index):
        self._save_relabel_result()
        self._index = index % len(self._find_idxs)
        self._render()

    def _render(self):
        from IPython.display import clear_output, display
        from bokeh.io.notebook import destroy_server
        from bokeh.io.state import curstate
        state = curstate()
        for server_id in list(state.uuid_to_server.keys()):
            destroy_server(server_id)
        new_btn_widgets = self._get_btn_widgets()
        clear_output()
        self._show_bokeh_bbox_edit_ui()
        display(new_btn_widgets)
