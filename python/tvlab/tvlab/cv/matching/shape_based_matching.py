import numpy as np
from tqdm.auto import trange

class ShapeBasedMatching:
    def __init__(self, features_num=128, T=(4,8),
                 weak_threshold=30.0, strong_threshold=60.0,
                 gaussion_kenel=7):
        '''
        yaml_dir (str): directory for save/load template info.
        features_num (int): number of features
        T (tuple): spred size on each pyramid level
        weak_threshold (float): magnitude threshold for get quantized angle
        strong_threshold (float): magnitude threshold for extract template
        gaussion_kenel (int): for blur input image
        '''
        from .impl.cshape_based_matching import CShapeBasedMatching
        self.T = T
        self.max_align = max(8, max([(i+1)*t*2 for i, t in enumerate(T)]))
        self.c = CShapeBasedMatching(features_num=features_num, T=T,
                                     weak_threshold=weak_threshold,
                                     strong_threshold=strong_threshold,
                                     gaussion_kenel=gaussion_kenel)

    def add(self, img, mask=None, class_id='default', angle_range=None, scale_range=None, pad=100):
        ''' add template
        img (np.ndarray): (H,W,C) or (H,W)
        mask (np.ndarray np.uint8): (H, W), 255 for object area, 0 for backgroud area
        class_id (str):
        angle_range (tuple): (start, stop, step) eg: (0, 360, 1)
        scale_range (tuple): (start, stop, step) eg: (0.1, 1.0, 0.01)
        pad (int): padding to avoid rotating out
        '''
        if angle_range is None:
            angle_range = (0.0, 0.0, 1.0)

        if scale_range is None:
            scale_range = (1.0, 1.0, 0.01)

        if img.ndim == 2:
            img = img[:,:,None]

        h, w, c = img.shape
        pad_img = np.zeros((h+2*pad, w+2*pad, c), dtype=np.uint8)
        pad_img[pad:pad+h, pad:pad+w] = img

        pad_mask = np.zeros((h+2*pad, w+2*pad), dtype=np.uint8)
        if mask is None:
            pad_mask[pad:pad+h, pad:pad+w] = 255
        else:
            pad_mask[pad:pad+h, pad:pad+w] = mask

        progress_bar = []
        def step_callback(i, n, pbar=progress_bar):
            try:
                if not progress_bar:
                    progress_bar.append(trange(n))
                progress_bar[0].update()
                return 0
            except Exception as e:
                print(e)
                return -1

        self.c.add(pad_img, pad_mask, class_id=class_id,
                   angle_range=angle_range,
                   scale_range=scale_range,
                   step_cb=step_callback)
        for pbar in progress_bar:
            pbar.close()

    def show(self, img, class_id='default', template_id=0, pad=100):
        ''' show template
        class_id (str):
        template_id (int):
        '''
        if img.ndim == 2:
            img = img[:,:,None]

        h, w, c = img.shape
        to_show = np.zeros((h+2*pad, w+2*pad, 3), dtype=np.uint8)
        to_show[pad:pad+h, pad:pad+w] = img

        ret = self.c.show(to_show, class_id=class_id, template_id=template_id)
        if ret == 0:
            return to_show
        print('template not found, add/load it first!.')
        return None

    def save(self, yaml_dir):
        '''
        yaml_dir(str): directory for save templates
        '''
        self.c.save(yaml_dir)

    def load(self, yaml_dir, class_ids="default"):
        '''
        yaml_dir(str): directory for load templates
        class_ids(list or str): load one or multi template
            eg: ['a', 'b', 'c'] or 'a'
        '''
        if isinstance(class_ids, (list, tuple)):
            class_ids = ','.join(class_ids)
        self.c.load(yaml_dir, class_ids)

    def find(self, img, score_threshold=90, iou_threshold=0.5, class_ids='default',
             pad=0, topk=-1, subpixel=False, debug=False):
        '''
        In:
            img (np.ndarray): (H,W,C) or (H,W)
            score_threshold (int): threshold for match score (0~100)
            iou_threshold(float): iou threshold for nms.
            class_ids(list or str): load one or multi template
                            eg: ['a', 'b', 'c'] or 'a'
            pad (int): padding for find object part is out of image
            topk (int): only keep topk result.
            subpixel (bool): Do subpixel and icp for get more accurate result.
        Out:
            matches (dict): {class_id0: [(x, y, w, h, angle, scale, score), ...], ..}
                    score: [0 ~ 100]
        '''
        if img.ndim == 2:
            img = img[:,:,None]

        h, w, c = img.shape

        pad_h = self.max_align*((h+2*pad)//self.max_align + 1)
        pad_w = self.max_align*((w+2*pad)//self.max_align + 1)

        pad_img = np.zeros((pad_h, pad_w, c), dtype=np.uint8)
        pad_img[pad:pad+h, pad:pad+w] = img

        debug_img = None
        if debug:
            debug_img = np.empty((pad_h, pad_w, 3), np.uint8)
            debug_img[:] = pad_img

        if isinstance(class_ids, (list, tuple)):
            class_ids = ','.join(class_ids)
        matches_arr, matches_ids = self.c.find(pad_img, debug_img, threshold=score_threshold,
                                               iou_threshold=iou_threshold, class_ids=class_ids,
                                               topk=topk, subpixel=subpixel)
        matches_result = None
        if matches_ids:
            matches_ids = matches_ids.split(',')
            class_ids = set(matches_ids)
            matches_result  = {c:[] for c in class_ids}
            assert len(matches_ids) == len(matches_arr)
            matches_arr[:, :2] -= pad
            for match_id, match in zip(matches_ids, matches_arr):
                matches_result[match_id].append(match.tolist())
        if debug:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(debug_img)
        return matches_result

    def draw_match_rect(self, img, matches=None, color='b', thickness=3, pad=0, alpha=0.5):
        '''
        img:
        matches: [(x,y,w,h,angle,scale,score), ...]
        '''
        from ..geometry import Region
        from tvlab.utils.basic import draw_polygons_on_img_pro

        if img.ndim == 2:
            img = img[:,:,None]

        h, w, c = img.shape
        to_show = np.zeros((h+2*pad, w+2*pad, 3), dtype=np.uint8)
        to_show[pad:pad+h, pad:pad+w] = img

        polygons = []
        labels = []
        for match in matches:
            x,y,w,h,angle,scale,score = match
            x += pad
            y += pad
            polygons.append(Region.from_rot_bbox((x,y, w, h, angle)).to_polygon())
            labels.append('angle:{:.1f}, scale:{:.2f}, score:{:.1f}'.format(angle, scale, score))
        to_show = draw_polygons_on_img_pro(to_show, polygons, labels, [color]*len(polygons), alpha=0.5)
        return to_show
