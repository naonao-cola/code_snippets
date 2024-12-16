'''
Copyright (C) 2023 TuringVision

Jupyter tool for image view and relabeling
'''
import json
import PIL
import os.path as osp
from io import BytesIO
import numpy as np

__all__ = ['ImageCleaner']


def _numpy_to_img(img_arr, format, img_quality):
    with BytesIO() as str_buf:
        PIL.Image.fromarray(img_arr).save(str_buf, format=format, quality=img_quality)
        return str_buf.getvalue()


class ImageCleaner:
    ''' Displays images for relabeling or deletion and saves changes in `json_path`.
    '''

    def __init__(self, ill, find_idxs=None, desc_list=None,
                 json_path=None, recovery=False,
                 debug_tfms=None,
                 ncols=4, nrows=2, labelset=None,
                 img_format='jpeg',
                 img_quality=95):
        '''
        # Arguments:
            ill: (ImageLabelList)
            find_idxs: (list) index in ill for relabeling
            desc_list: (list) description of each find item
            json_path: (str) output json path for save changes
            recovery: (bool) Is recovery relabel info from json_path?
            debug_tfms: (list) eg: [train_tfms, valid_tfms]
            ncols: (int) number of column
            nrows: (int) number of row
            labelset: (list) list of label, get from ill when it's None
            img_format: one of 'jpeg', 'png', 'bmp'
                    'jpeg' for high speed browse image,
                    'png' and 'bmp' for high resolution browse image.
        '''
        self._ill = ill
        self._deleted_info = set()
        self._relabel_info = None
        self._img_format = img_format
        self._img_quality = img_quality
        self._find_idxs = find_idxs
        self._desc_list = desc_list
        self._json_path = json_path
        self._labelset = ill.labelset() if labelset is None else labelset
        if not self._labelset:
            self._labelset = ['object']
        self._ncols = ncols
        self._nrows = nrows
        self._index = 0
        if recovery:
            self._load_json()
        if self._find_idxs is None:
            self._find_idxs = list(range(len(ill)))
        self._debug_tfms = debug_tfms
        self._update_json()
        self._render()

    @classmethod
    def find_toplosses_idxs(cls, ill, losses, n_imgs=None):
        import torch
        if not n_imgs: n_imgs = len(ill)
        idxs = torch.topk(losses, n_imgs)[1]
        return idxs.tolist()

    @classmethod
    def from_toplosses(cls, ill, losses, n_imgs=None, **kwargs):
        idxs = cls.find_toplosses_idxs(ill, losses, n_imgs)
        return cls(ill, idxs, **kwargs)

    @classmethod
    def make_button_widget(cls, label, handler=None, style=None, layout=None):
        "Return a Button widget with specified `handler`."
        from ipywidgets import widgets
        btn = widgets.Button(description=label, layout=layout)
        if handler is not None: btn.on_click(handler)
        if style is not None: btn.button_style = style
        return btn

    @classmethod
    def make_dropdown_widget(cls, description='Description', options=['Label 1', 'Label 2'],
                             value='Label 1',
                             layout=None, handler=None):
        "Return a Dropdown widget with specified `handler`."
        from ipywidgets import widgets
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        if handler is not None: dd.observe(handler, names=['value'])
        return dd

    def _relabel(self, change):
        class_new, index = change.new, change.owner.img_index
        if change.owner.ori_label != class_new:
            change.owner.relabel_info[str(index)] = class_new
            change.owner.source_label_btn.button_style = 'danger'
        else:
            change.owner.relabel_info.pop(str(index))
            change.owner.source_label_btn.button_style = 'info'
        self._update_json()

    def _update_delete(self, btn):
        btn.button_style = "danger" if str(btn.img_index) in self._deleted_info else ""

    def _on_delete(self, btn):
        index = str(btn.img_index)
        if index in self._deleted_info:
            self._deleted_info.remove(index)
        else:
            self._deleted_info.add(index)
        self._update_json()
        self._update_delete(btn)

    def _get_label_widget(self, ori_label, index, labelset, relabel_info):
        from ipywidgets import widgets, Layout
        if ori_label is None:
            ori_label = 'None'

        label_widget_list = list()
        if isinstance(ori_label, str):
            label = relabel_info[str(index)] if str(index) in relabel_info else ori_label
            source_label = widgets.Button(description=ori_label, disabled=True,
                                          button_style='info' if label == ori_label else 'danger',
                                          layout=Layout(height='auto', width='auto'))
            label_widget_list.append(source_label)
            if len(labelset) > 1:
                labelset = labelset.copy()
                labelset.append('None')
                dropdown = self.make_dropdown_widget(description='', options=labelset,
                                                     value=label, handler=self._relabel,
                                                     layout=Layout(height='auto', width='auto'))
                dropdown.img_index = index
                dropdown.source_label_btn = source_label
                dropdown.ori_label = ori_label
                dropdown.relabel_info = relabel_info
                label_widget_list.append(dropdown)
        return label_widget_list

    def _get_img_widget(self, index, desc=None):
        from ipywidgets import widgets, Layout
        im, ori_label = self._ill[index]
        jpeg_im = _numpy_to_img(im, self._img_format, self._img_quality)
        img_widget = widgets.Image(value=jpeg_im, format=self._img_format,
                                   layout=Layout(height='auto', width='auto'))
        img_widgets = [img_widget]
        if self._debug_tfms:
            for tfm in self._debug_tfms:
                im, _ = self._ill.do_tfms([tfm], im, ori_label,
                                          index, self._ill.x[index])
                jpeg_im = _numpy_to_img(im, self._img_format, self._img_quality)
                img_widget = widgets.Image(value=jpeg_im, format=self._img_format,
                                           layout=Layout(height='auto', width='auto'))
                img_widgets.append(img_widget)

        delete_btn = self.make_button_widget('Delete',
                                             handler=self._on_delete,
                                             layout=Layout(height='auto', width='auto'))
        delete_btn.img_index = index
        self._update_delete(delete_btn)

        fdesc = str(index)
        if desc is not None:
            fdesc = fdesc + ': ' + desc

        desc_label = widgets.Button(description=fdesc, tooltip=fdesc, disabled=True,
                                    button_style='primary',
                                    layout=Layout(height='auto', width='auto'))

        if isinstance(ori_label, (list, tuple)):
            if self._relabel_info is None:
                self._relabel_info = [dict() for _ in ori_label]

            all_label_widget_list = list()
            for i, ori_label_one in enumerate(ori_label):
                label_widget_list = self._get_label_widget(ori_label_one, index,
                                                           self._labelset[i],
                                                           self._relabel_info[i])
                if i == 0:
                    label_widget_list.append(delete_btn)
                label_widget = widgets.HBox(label_widget_list)
                all_label_widget_list.append(label_widget)
            all_label_widget_list.append(desc_label)
            label_widget = widgets.VBox(all_label_widget_list)
        else:
            if self._relabel_info is None:
                self._relabel_info = dict()
            label_widget_list = self._get_label_widget(ori_label, index,
                                                       self._labelset, self._relabel_info)
            label_widget_list.append(delete_btn)
            label_widget = widgets.HBox(label_widget_list)
            label_widget = widgets.VBox([label_widget, desc_label])

        return widgets.VBox([*img_widgets, label_widget],
                            layout=Layout(overflow="hidden",
                                          width=format(1 / self._ncols, '.1%'),
                                          height=format(1 / self._nrows, '.1%')))

    def _get_img_widgets(self):
        from ipywidgets import widgets
        img_widgets = []
        for y in range(self._nrows):
            x_widgets = []
            for x in range(self._ncols):
                i = y * self._ncols + x + self._index
                if i >= len(self._find_idxs):
                    continue
                desc = self._desc_list[i] if self._desc_list is not None else None
                index = self._find_idxs[i]
                x_widgets.append(self._get_img_widget(index, desc))
            img_widgets.append(widgets.HBox(x_widgets))

        return widgets.VBox(img_widgets)

    def _update_index(self, index):
        self._index = index % len(self._find_idxs)
        self._render()

    def _update_step(self, step):
        index = self._index + step
        self._update_index(index)

    def _add_label(self, widget):
        try:
            add_label_list = eval(widget.value)
            if isinstance(self._labelset[0], list) and isinstance(add_label_list[0], list):
                for i, add_label in enumerate(add_label_list):
                    self._labelset[i] += add_label
            elif isinstance(self._labelset[0], str) and isinstance(add_label_list[0], str):
                self._labelset += add_label_list
            widget.value = ''
            self._update_index(self._index)
        except Exception as e:
            pass

    def _get_btn_widgets(self):
        from ipywidgets import widgets, Layout
        batch = self._ncols * self._nrows
        half_batch = int(batch / 2)
        prev_btn = self.make_button_widget('Prev batch', style="info",
                                           handler=lambda _: self._update_step(-batch),
                                           layout=Layout(height='auto', width='100px'))
        half_prev_btn = self.make_button_widget('Prev', style="info",
                                                handler=lambda _: self._update_step(-half_batch),
                                                layout=Layout(height='auto', width='60px'))
        half_next_btn = self.make_button_widget('Next', style="primary",
                                                handler=lambda _: self._update_step(half_batch),
                                                layout=Layout(height='auto', width='60px'))
        next_btn = self.make_button_widget('Next batch', style="primary",
                                           handler=lambda _: self._update_step(batch),
                                           layout=Layout(height='auto', width='100px'))

        progress_bar = widgets.IntSlider(self._index, 0, len(self._find_idxs) - 1, step=batch,
                                         description=' Index:',
                                         continuous_update=False,
                                         readout=True,
                                         readout_format='d',
                                         layout=Layout(width='40%'))
        progress_bar.observe(lambda change: self._update_index(change['new']), names='value')
        progress_txt = widgets.Label('Total: {}'.format(len(self._find_idxs)),
                                     layout=Layout(width='10%'))

        add_label_placeholder = "add label eg: ['dot', 'cat']"
        if isinstance(self._labelset[0], list):
            add_label_placeholder = "add label eg: [['mcode1'], ['subc1']]"

        add_label = widgets.Text('', placeholder=add_label_placeholder,
                                 description='',
                                 layout=Layout(width='200px'),
                                 disabled=False)
        add_label.on_submit(self._add_label)
        hbox = widgets.HBox([prev_btn, half_prev_btn, half_next_btn, next_btn,
                             progress_bar, progress_txt, add_label])
        return hbox

    def get_info(self):
        ''' get relabel info
        '''
        return {'index': self._index,
                'relabel': self._relabel_info,
                'delete': list(self._deleted_info),
                'desc_list': self._desc_list,
                'find_idxs': self._find_idxs,
                'image_list': self._ill.x}

    def get_ill(self):
        '''get modified ill
        '''
        from ..category.multi_task_image_data import ImageMultiLabelList
        new_ill = self._ill.copy()
        if isinstance(self._ill, ImageMultiLabelList):
            for li, ri in enumerate(self._relabel_info):
                for idx, lb in ri.items():
                    new_ill.y[int(idx)][li] = lb
        else:
            for l in self._relabel_info:
                new_ill.y[int(l)] = self._relabel_info[l]

        count = 0
        new_deleted_info = np.array(list(map(int, self._deleted_info)))
        new_deleted_info.sort()
        for i in new_deleted_info:
            del new_ill.y[i - count]
            del new_ill.x[i - count]
            count += 1

        return new_ill

    def to_json(self, json_path):
        ''' dump relabel info to json
        '''
        with open(json_path, 'wt', encoding='utf-8') as fp:
            json.dump(self.get_info(), fp)

    def _update_json(self):
        if self._json_path:
            self.to_json(self._json_path)

    def _load_json(self):
        if self._json_path and osp.isfile(self._json_path):
            with open(self._json_path, 'rt', encoding='utf-8') as fp:
                info = json.load(fp)
                if 'relabel' in info:
                    self._relabel_info = info['relabel']
                if 'delete' in info:
                    self._deleted_info = set(info['delete'])
                if 'index' in info:
                    self._index = info['index']
                if 'desc_list' in info and self._desc_list is None:
                    self._desc_list = info['desc_list']
                if 'find_idxs' in info and self._find_idxs is None:
                    self._find_idxs = info['find_idxs']

    def _render(self):
        from IPython.display import clear_output, display
        new_img_widgets = self._get_img_widgets()
        new_btn_widgets = self._get_btn_widgets()
        clear_output()
        display(new_img_widgets)
        display(new_btn_widgets)
