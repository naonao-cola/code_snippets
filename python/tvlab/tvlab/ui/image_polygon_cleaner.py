'''
Copyright (C) 2023 TuringVision

Jupyter tool for image with polygons view and relabeling
'''
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
from ..utils import get_notebook_url
from .image_cleaner import ImageCleaner


__all__ = ['ImagePolygonCleaner']


def _y2source(y, label_level=False):
    from bokeh.models import ColumnDataSource
    if not y or 'labels' not in y or 'polygons' not in y:
        y = {'labels': list(), 'polygons': list(), 'colors': list()}
    polygons = y['polygons']
    labels = y['labels']
    colors = y.get('colors', ['springgreen'] * len(labels))
    levels = y.get('levels', [5] * len(labels))
    x, y = list(), list()
    l, t = list(), list()
    text = list()
    color = list()
    for polygon, label, c, level in zip(polygons, labels, colors, levels):
        if len(polygon) % 2 == 1:
            polygon = polygon[:-1]
        polygon = np.array(polygon).reshape(-1, 2)
        _x = polygon[:, 0]
        _y = polygon[:, 1]
        x.append(_x)
        y.append(_y)
        _max_y_i = np.argmin(_y)
        t.append(_y[_max_y_i])
        l.append(_x[_max_y_i])
        if label_level:
            label = label + '-' + str(level)
        text.append(label)
        color.append(c)

    source = ColumnDataSource(dict(x=x, y=y, l=l, t=t, text=text, color=color))
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
    x = source.data['x']
    y = source.data['y']
    colors = source.data['color']
    polygons = []
    for _x, _y in zip(x, y):
        polygon = []
        if isinstance(_x, dict):
            keys = sorted([int(i) for i in _x.keys()])
            for k in keys:
                k = str(k)
                polygon.append(_x[k])
                polygon.append(_y[k])
        else:
            for _ix, _iy in zip(_x, _y):
                polygon.append(_ix)
                polygon.append(_iy)
        polygons.append(polygon)

    y = {'labels': labels, 'polygons': polygons, 'colors': colors}
    if label_level:
        y['levels'] = levels
    return y


def _check_y_change(y1, y2, label_level=False):
    if y1['labels'] != y2['labels'] or y1['polygons'] != y2['polygons']:
        return True
    if label_level:
        if 'levels' not in y1:
            return True
        if 'levels' in y1 and 'levels' in y2 and y1['levels'] != y2['levels']:
            return True
    return False


class ImagePolygonCleaner(ImageCleaner):
    #for speedup bokeh show image
    IMG_CACHE_DIR = 'img_polygon_cleaner_cache'

    '''Graphical image annotation tool for instance segmentation task.
    '''
    def __init__(self, ipll, find_idxs=None,
                 desc_list=None,
                 notebook_url=None,
                 ncols=2, nrows=1, labelset=None,
                 polygon_tfm=None,
                 max_size=600,
                 json_dir=None,
                 active_zoom=False,
                 share_xy=False,
                 label_level=None,
                 img_format='jpeg',
                 img_quality=95,
                 port=None):
        '''
        # Arguments:
            ipll: (ImagePolygonLabelList)
            find_idxs: (list) index in ill for relabeling
            desc_list: (list) description of each find item
            notebook_url: (str) 'ipaddr:port' for bokeh data sync in notebook
                        or call set_notebook_url('ipaddr:port') first
            ncols: (int) number of column
            nrows: (int) number of row
            labelset: (list) list of label, get from ill when it's None
            polygon_tfm: transform function for restore polygon before save change.
                Default will rescale the polygon to the original image size if
                polygon_tfm is None.
                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bpolygon by adding a `polygon_tfm` function.

            json_dir: (str) json (labelme format) output path (only save changed image)
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

        self._ipll = ipll
        self._new_ipll = ipll.copy()
        self._relabel_info = None
        self._img_format = img_format
        self._img_quality = img_quality
        self._find_idxs = find_idxs
        self._desc_list = desc_list
        self._notebook_url = notebook_url
        self._bokeh_port = port
        self._img_cache_dir = ImagePolygonCleaner.IMG_CACHE_DIR
        self._labelset = ipll.labelset() if labelset is None else labelset
        self._active_zoom = active_zoom
        self._share_xy = share_xy
        if label_level is None:
            label_level = True if 'levels' in ipll.y[0] else False
        self._label_level = label_level
        if not self._labelset:
            self._labelset = ['object']

        self._polygon_tfm = polygon_tfm
        self._ncols = ncols
        self._nrows = nrows
        self._max_size = max_size
        self._json_dir = json_dir
        # for avoid image_url cache
        self._rand_key = str(np.random.randint(0, 1e8))
        self._op_id = 0
        self._index = 0
        if self._find_idxs is None:
            self._find_idxs = list(range(len(ipll)))
        self._source_list = None
        self._old_y_list = None
        self._scale_ratio_list = None
        self._ori_image_shape_list = None
        self._render()

    @property
    def find_idxs(self):
        return self._find_idxs

    def get_ipll(self):
        ''' get modified ipll
        '''
        self._save_relabel_result()
        new_ipll = self._new_ipll.copy()
        new_ipll.y = new_ipll.y.filter_by_label(lambda l: l in self._labelset)
        return new_ipll

    def get_info(self):
        raise NotImplementedError

    def to_json(self, json_path):
        raise NotImplementedError

    def _clean_cache_dir(self):
        import shutil
        shutil.rmtree(self._img_cache_dir, ignore_errors=True)
        os.makedirs(self._img_cache_dir, exist_ok=True)

    def _show_bokeh_polygon_edit_ui(self):
        from bokeh.io import show
        from .bokeh_polygon_edit_tool import get_bokeh_polygon_edit_app
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
                img, y = self._new_ipll[index]
                img_path = self._new_ipll.x[index]
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

        bk_app = get_bokeh_polygon_edit_app(img_path_list,
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
        from ..segmentation.polygon_label import PolygonLabel

        def _scale_polygon_tfm(polygon, scale_ratio):
            if len(polygon) %2 == 1:
                polygon = polygon[:-1]
            polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            polygon[:, 0] *= scale_ratio[1]
            polygon[:, 1] *= scale_ratio[0]
            return polygon.flatten().tolist()

        if self._source_list:
            change_x, change_y = [], []
            for i, source in enumerate(self._source_list):
                index = self._find_idxs[i+self._index]
                old_y = self._old_y_list[i]
                new_y = _source2y(source, self._label_level)
                old_y_gt = PolygonLabel(old_y).filter_by_label(lambda l: l in self._labelset)
                new_y_gt = PolygonLabel(new_y).filter_by_label(lambda l: l in self._labelset)
                scale_ratio = self._scale_ratio_list[i]
                ori_shape = self._ori_image_shape_list[i]
                if _check_y_change(old_y_gt, new_y_gt, self._label_level):
                    if self._polygon_tfm:
                        new_y['polygons'] = [self._polygon_tfm(poly, ori_shape)
                                             for poly in new_y['polygons']]
                    else:
                        new_y['polygons'] = [_scale_polygon_tfm(poly, scale_ratio)
                                             for poly in new_y['polygons']]
                    self._new_ipll.y[index] = new_y
                    change_x.append(self._new_ipll.x[index])
                    change_y.append(new_y_gt)
            if self._json_dir and change_x:
                from ..segmentation import ImagePolygonLabelList
                _ipll = ImagePolygonLabelList(change_x, change_y)
                _ipll.to_labelme(self._json_dir)

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
        self._show_bokeh_polygon_edit_ui()
        display(new_btn_widgets)
