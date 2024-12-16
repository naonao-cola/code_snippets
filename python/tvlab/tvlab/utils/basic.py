'''
Copyright (C) 2023 TuringVision

utils
'''
import os
import json
import yaml
import pickle
import cv2
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count()//2)


def in_notebook():
    "Check if the code is running in a jupyter notebook"
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell': # Jupyter notebook, Spyder or qtconsole
            import IPython
            #IPython version lower then 6.0.0 don't work with output you update
            return IPython.__version__ >= '6.0.0'
        elif shell == 'TerminalInteractiveShell': return False  # Terminal running IPython
        else: return False  # Other type (?)
    except NameError: return False      # Probably standard Python interpreter


IN_NOTEBOOK = in_notebook()


__all__ = ['cv2', 'np', 'plt', 'os', 'osp', 'trange', 'tqdm',
           'safe_div', 'thread_pool', 'obj_to_json', 'obj_from_json',
           'obj_to_yaml', 'obj_from_yaml',
           'obj_to_pkl', 'obj_from_pkl', 'draw_bboxes_on_img',
           'draw_polygons_on_img_pro', 'draw_bboxes_on_img_pro',
           'least_common_multiple', 'kfold_split', 'polygon_to_bbox',
           'set_gpu_visible', 'set_notebook_url', 'get_notebook_url',
           'dump_cuda_mem', 'mask_to_polygon', 'mask_to_polygons',
           'Image', 'seed_everything', 'path_compare', 'IN_NOTEBOOK', 'img_label_path_match']


def seed_everything(seed: int):
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(x, y):
    if y == 0:
        return 0
    return x / y


def polygon_to_bbox(polygon):
    if len(polygon)%2 == 1:
        polygon = polygon[:-1]
    polygon = np.array(polygon).reshape(-1, 2)
    x = polygon[:, 0]
    y = polygon[:, 1]
    l = min(x)
    r = max(x)
    t = min(y)
    b = max(y)
    return [l,t,r,b]


def dump_cuda_mem():
    import torch
    mem_allocated = torch.cuda.memory_allocated()/(1024**3)
    max_mem_allocated = torch.cuda.max_memory_allocated()/(1024**3)
    print('GPU mem allocated: {:.1f}G, max allocated: {:.1f}G'.format(mem_allocated,
                                                                      max_mem_allocated))
    mem_cached = torch.cuda.memory_cached()/(1024**3)
    max_mem_cached = torch.cuda.max_memory_cached()/(1024**3)
    print('GPU mem cached: {:.1f}G, max cached: {:.1f}G'.format(mem_cached,
                                                                max_mem_cached))


def set_gpu_visible(devices):
    '''
    devices: int or str
        0 or '0,1,2,3'

    Must be called before calling `import torch`.
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def set_notebook_url(url):
    '''url: '172.0.0.1:8888'
    '''
    os.environ["NOTEBOOK_URL"] = url

def get_notebook_url():
    url = None
    try:
        url = os.environ["NOTEBOOK_URL"]
    except KeyError:
        pass
    return url

def kfold_split(idxs, fold, slice_num, valid_slice_num):
    one_fold_step = slice_num // fold
    rand_idx_slice = [idxs[i::slice_num] for i in range(slice_num)]
    k_fold_train_idx = list()
    k_fold_valid_idx = list()
    for f in range(fold):
        start_i = f * one_fold_step
        end_i = start_i + valid_slice_num
        end_i = end_i % slice_num
        if start_i < end_i:
            valid_slice_idx = rand_idx_slice[start_i:end_i]
            train_slice_idx = rand_idx_slice[:start_i] + rand_idx_slice[end_i:]
        else:
            valid_slice_idx = rand_idx_slice[:end_i] + rand_idx_slice[start_i:]
            train_slice_idx = rand_idx_slice[end_i:start_i]
        train_idx = [i for idxs in train_slice_idx for i in idxs]
        valid_idx = [i for idxs in valid_slice_idx for i in idxs]
        k_fold_train_idx.append(train_idx)
        k_fold_valid_idx.append(valid_idx)
    return k_fold_train_idx, k_fold_valid_idx


def least_common_multiple(a, b):
    a, b = int(a), int(b)
    s = a * b
    while a % b != 0:
        a, b = b, (a % b)
    return s // b


def _worker_func(func, items, idxs, cb):
    for i in idxs:
        try:
            item = items[i]
            if func:
                func(item)
            cb(i)
        except Exception as e:
            print(e)


def thread_pool(func, items, workers):
    '''
    func: input one item of items. Tips: use functools.partial for fixed parameters.
        def task_func(item):
            pass
    items: Iterator
    workers: number of worker
    '''
    import concurrent.futures
    from tqdm.auto import trange

    task_idxs = list(range(len(items)))
    worker_task_idxs = [task_idxs[i::workers] for i in range(workers)]
    with trange(len(items)) as t:
        def update_cb(i):
            task_idxs.remove(i)
            t.update()

        with concurrent.futures.ThreadPoolExecutor(workers) as e:
            for i in range(workers):
                e.submit(_worker_func, func, items, worker_task_idxs[i], update_cb)

        # retry for failed item
        if task_idxs:
            _worker_func(func, items, task_idxs[:], update_cb)
    return task_idxs


def obj_to_json(obj, json_path, ensure_ascii=True):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'wt', encoding='utf-8') as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=ensure_ascii)


def obj_from_json(json_path):
    with open(json_path, 'rt', encoding='utf-8') as fp:
        return json.load(fp)


def obj_to_yaml(obj, yaml_path):
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'wt', encoding='utf-8') as fp:
        yaml.dump(obj, fp)


def obj_from_yaml(yaml_path):
    with open(yaml_path, 'rt', encoding='utf-8') as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def obj_to_pkl(obj, pkl_path):
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as fp:
        pickle.dump(obj, fp)

def obj_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fp:
        return pickle.load(fp)


def draw_bboxes_on_img(img, bboxes, labels=None, color=(255, 0, 0),
                       font_scale=1, thickness=3, font_offset=(0, 0)):
    '''
    In:
        img: numpy image
        bboxes: [(l,t,r,b), ...]
        labels: ['a', 'b', ..]
        min_box_shape: (w,h)
    Out:
        img: numpy image
    '''
    import cv2

    if not bboxes:
        return img

    img = img.copy() # fix sametimes cv2 return cv2.UMat
    for i, bbox in enumerate(bboxes):
        l, t, r, b = [int(item) for item in bbox[:4]]
        img = cv2.rectangle(img, (l, t), (r, b), color=color, thickness=thickness)
        if labels:
            cv2.putText(img, labels[i], (l+font_offset[0], b+font_offset[1]),
                        cv2.FONT_ITALIC, font_scale, color, thickness)
    return img


def draw_polygons_on_img_pro(img, polygons, labels=None, colors=None, alpha=0.5, is_rect=False):
    '''
    In:
        img: numpy image
        polygons: [(x1, y1, x2, y2, x3, y3, ...), ...]
        labels: ['a', 'b', ..]
        colors: ['red', 'blue', ...]
        alpha (float): 0.5
    Out:
        img: numpy image
    '''
    import matplotlib as mpl
    import matplotlib.colors as mplc
    import matplotlib.figure as mplfigure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _SMALL_OBJECT_AREA_THRESH = 1000

    height, width = img.shape[:2]
    fig = mplfigure.Figure(frameon=False)
    dpi = fig.get_dpi()
    fig.set_size_inches(
            (width + 1e-2) / dpi,
            (height + 1e-2) / dpi,
        )
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.set_xlim(0.0, width)
    ax.set_ylim(height)

    default_font_size = max(np.sqrt(height * width) // 90, 10)


    def _change_color_brightness(color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        import colorsys
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    for i, polygon in enumerate(polygons):
        color = 'gold'
        if colors:
            color = colors[i]
        if is_rect:
            box = polygon
            x0, y0, x1, y1 = box[:4]
            box_w = x1 - x0
            box_h = y1 - y0
            linewidth = max(default_font_size / 4, 1)
            patch = mpl.patches.Rectangle(
                (x0, y0),
                box_w,
                box_h,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle='-',
            )
        else:
            if len(polygon)%2 == 1:
                polygon = polygon[:-1]

            box = polygon_to_bbox(polygon)
            x0, y0, x1, y1 = box[:4]
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = _change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
                edge_color = mplc.to_rgb(edge_color) + (1,)

            polygon = np.array(polygon).reshape(-1, 2)
            patch = mpl.patches.Polygon(
                polygon,
                fill=True,
                facecolor=mplc.to_rgb(color) + (alpha,),
                edgecolor=edge_color,
                linewidth=max(default_font_size // 15, 1),
            )
        ax.add_patch(patch)

        if labels:
            text = labels[i]

            horizontal_alignment = "left"
            instance_area = (y1 - y0) * (x1 - x0)
            text_pos = (x0, y0)
            if (instance_area < _SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40):
                if y1 >= height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            height_ratio = (y1 - y0) / np.sqrt(height * width)
            lighter_color = _change_color_brightness(color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * default_font_size
            )

            lighter_color = np.maximum(list(mplc.to_rgb(lighter_color)), 0.2)
            lighter_color[np.argmax(lighter_color)] = max(0.8, np.max(lighter_color))

            ax.text(
                text_pos[0],
                text_pos[1],
                text,
                size=font_size,
                family="sans-serif",
                bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=lighter_color,
                zorder=10,
                rotation=0,
            )

    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype="uint8")
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)

    alpha = alpha.astype("float32") / 255.0
    if len(img.shape) == 2:
        img = img[:,:,None]
    visualized_image = img * (1 - alpha) + rgb * alpha

    visualized_image = visualized_image.astype("uint8")
    return visualized_image


def draw_bboxes_on_img_pro(img, bboxes, labels=None, colors=None, alpha=0.5):
    '''
    In:
        img: numpy image
        bboxes: [(l,t,r,b), ...]
        labels: ['a', 'b', ..]
        colors: ['red', 'blue', ...]
        alpha (float): 0.5
    Out:
        img: numpy image
    '''
    return draw_polygons_on_img_pro(img, bboxes, labels=labels,
                                   colors=colors, alpha=alpha, is_rect=True)


def cnt_to_polygon(cnt):
    from shapely import geometry
    from shapely.geometry import Polygon
    try:
        poly = Polygon(cnt.reshape(-1, 2))
        poly = poly.buffer(1)
    except ValueError:
        return []

    ignore_types = (geometry.LineString,
                    geometry.MultiLineString,
                    geometry.point.Point,
                    geometry.MultiPoint,
                    geometry.GeometryCollection)
    if isinstance(poly, geometry.Polygon):
        poly = geometry.MultiPolygon(
            [poly])
    elif isinstance(poly,
                    geometry.MultiPolygon):
        pass
    elif isinstance(poly, ignore_types):
        # polygons that become (one or more) lines/points after clipping
        # are here ignored
        poly = geometry.MultiPolygon([])
    else:
        raise Exception(
            "Got an unexpected result of type %s from Shapely for "
            "image (%d, %d) and polygon %s. This is an internal error. "
            "Please report." % (type(poly))
        )

    max_area = 0
    max_poly = None
    for poly_inter_shapely in poly.geoms:
        area = poly_inter_shapely.area
        if area > max_area:
            max_area = area
            max_poly = poly_inter_shapely

    poly = [i for xy in list(max_poly.exterior.coords) for i in xy]
    return poly


def mask_to_polygon(mask):
    '''
    In:
        mask (np.ndarray): (HxW)

    Out:
        [x1, y1, x2, y2, ...]
    '''
    mask = mask.astype(np.uint8)
    ret = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = ret[0] if len(ret) != 3 else ret[1]
    if not contours:
        return None
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    cnt = contours[areas.argmax()]
    poly = cnt_to_polygon(cnt)
    return poly


def mask_to_polygons(mask, area_threshold=25):
    '''
    In:
        mask (np.ndarray): (HxW)

    Out:
        [[x1,y1,x2,y2,...], [x1,y1,x2,y2,...]
    '''
    mask = mask.astype(np.uint8)
    ret = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = ret[0] if len(ret) != 3 else ret[1]
    if not contours:
        return None
    areas = np.array([cv2.contourArea(cnt) for cnt in contours])
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > area_threshold:
            poly = cnt_to_polygon(cnt)
            polygons.append(poly)
    return polygons


def path_compare(path1, path2, reverse=False):
    if reverse:
        path1 = path1[::-1]
        path2 = path2[::-1]
    i = min(len(path1), len(path2))
    for j in range(i):
        if path1[j] != path2[j]:
            break
    return j


def img_label_path_match(img_dir, lbl_dir, ext='.xml', check_ext=True, recurse=True, followlinks=False):
    from ..category import get_image_files, get_files

    img_path_list = get_image_files(img_dir, check_ext=check_ext,
                                    recurse=recurse, followlinks=followlinks)
    lbl_path_list = get_files(lbl_dir, extensions=[ext],
                              recurse=recurse, followlinks=followlinks)
    img_name_list = [osp.splitext(osp.basename(img))[0] for img in img_path_list]
    lbl_name_list = [osp.splitext(osp.basename(lbl))[0] for lbl in lbl_path_list]

    label_info = list()
    label_set = set()
    match_result = dict()
    for img_path in img_path_list:
        img_name = osp.splitext(osp.basename(img_path))[0]
        match_cnt = lbl_name_list.count(img_name)
        lbl_path = None
        # check one img name match multi lbl
        if match_cnt == 1:
            lbl_path = lbl_path_list[lbl_name_list.index(img_name)]
        elif match_cnt > 1:
            all_find_lbl = [lbl_path_list[i] for i, lbl_name in enumerate(lbl_name_list)
                            if lbl_name == img_name]
            max_s = 0
            for find_lbl_path in all_find_lbl:
                lbl_path_suffix = osp.dirname(find_lbl_path[len(lbl_dir):])
                img_path_suffix = osp.dirname(img_path[len(img_dir):])
                s = path_compare(lbl_path_suffix, img_path_suffix)
                if s > max_s:
                    lbl_path = find_lbl_path
                    max_s = s

        # check one lbl name match multi img
        if lbl_path is not None:
            match_cnt = img_name_list.count(img_name)
            if match_cnt > 1:
                all_find_img = [img_path_list[i] for i, _name in enumerate(img_name_list)
                                if _name == img_name]
                max_s = 0
                match_img_path = None
                for find_img_path in all_find_img:
                    lbl_path_suffix = osp.dirname(lbl_path[len(lbl_dir):])
                    img_path_suffix = osp.dirname(find_img_path[len(img_dir):])
                    s = path_compare(lbl_path_suffix, img_path_suffix)
                    if s > max_s:
                        match_img_path = find_img_path
                        max_s = s
                if match_img_path != img_path:
                    lbl_path = None
        match_result[img_path] = lbl_path
    return match_result


