'''
Copyright (C) 2019 ThunderSoft

Template matching defect detector
'''
import numpy as np
from ...cv.filter import peak_local_max
from ...cv.basic import ncc


def incremental_stats_estimation(image_stack_t, iteration_count=1):
    import torch
    weights = torch.ones_like(image_stack_t)

    mean_t = torch.mean(image_stack_t, dim=0)
    variance_t = torch.var(image_stack_t, dim=0)
    diff_2_t = (image_stack_t - mean_t) ** 2

    for i in range(iteration_count):
        weights = torch.exp(-(diff_2_t/(4*torch.clamp_min(variance_t, 1e-7))))
        mean_t = torch.sum(image_stack_t * weights, dim=0) / torch.sum(weights, dim=0)
        diff_2_t = (image_stack_t - mean_t) ** 2
        variance_t = torch.sum(diff_2_t * weights, dim=0) / torch.sum(weights, dim=0)
    std_t = torch.sqrt(variance_t)
    std_t = std_t.clamp_min(1e-3)
    return mean_t, std_t


def get_saliency_map(img_t, pattern_shape=(64, 64), stride=(32, 32),
                     threshold_low=0.8,
                     threshold_high=0.9,
                     max_similar_cnt=8, **kwargs):
    '''
    img_t: (H,W)
    '''
    import torch

    img_h, img_w = img_t.shape[:2]

    pattern_h, pattern_w = pattern_shape
    stride_h, stride_w = stride

    x_strides = np.linspace(0, img_w-pattern_w, img_w//stride_w, dtype=np.int)
    y_strides = np.linspace(0, img_h-pattern_h, img_h//stride_h, dtype=np.int)

    img_t = img_t.type(torch.float32)
    # HxW -> (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x pattern_h x pattern_w
    img_unfold_t = img_t.unfold(0, pattern_h, 1).unfold(1, pattern_w, 1)

    # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h) x (pattern_w)
    # --->
    # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h * pattern_w)
    img_unfold_t = img_unfold_t.reshape([img_unfold_t.shape[0], img_unfold_t.shape[1], -1])

    unfold_h, unfold_w = img_unfold_t.shape[:2]

    pick_unfold_idx = [y*unfold_w+x for y in y_strides for x in x_strides]

    # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h * pattern_w)
    # --->
    # ((img_h - pattern_h + 1)*(img_w - pattern_w + 1)) x (pattern_h * pattern_w)
    img_unfold_t = img_unfold_t.reshape([img_unfold_t.shape[0]*img_unfold_t.shape[1], -1])
    pick_img_unfold_t = img_unfold_t[pick_unfold_idx]

    # get template match score
    # scores: (len(y_strides) * len(x_strides)) x (unfold_h * unfold_w)
    scores = ncc(pick_img_unfold_t, img_unfold_t)

    scores = scores.reshape(-1, unfold_h, unfold_w)
    zyx_list = peak_local_max(scores, min_distance=(pattern_h//2, pattern_w//2),
                              threshold_low=threshold_low,
                              threshold_high=threshold_high,
                              indices=True, num_peaks=max_similar_cnt)

    max_scores, _ = scores.reshape(scores.shape[0], -1).max(dim=1)

    sum_t = torch.full_like(img_t, 0.0, dtype=torch.float32, device=img_t.device)
    weight_t = torch.full_like(img_t, 0.0, dtype=torch.float32, device=img_t.device)
    std_t = torch.full_like(img_t, 255.0, dtype=torch.float32, device=img_t.device)

    #for i, score in enumerate(scores):
    for i, yx_list in enumerate(zyx_list):
        idx = pick_unfold_idx[i]
        ori_x = idx % unfold_w
        ori_y = idx // unfold_w

        if (ori_y, ori_x) not in yx_list:
            yx_list.append((ori_y, ori_x))

        # get similar pattern

        if len(yx_list) > 1:
            is_border = (ori_x == x_strides[0] or ori_x == x_strides[-1] or
                         ori_y == y_strides[0] or ori_y == y_strides[-1])
            use_yx_list = [(ori_y, ori_x)]
            if is_border:
                use_yx_list = yx_list

            for y, x in use_yx_list:
                pattern_stack_list = list()
                for _y, _x in yx_list:
                    if (_y, _x) != (y, x):
                        pattern = img_t[_y:_y+pattern_h, _x:_x+pattern_w]
                        pattern_stack_list.append(pattern)

                pattern_stacks = torch.stack(pattern_stack_list)
                sum_t[y:y+pattern_h, x:x+pattern_w] += pattern_stacks.sum(dim=0)
                weight_t[y:y+pattern_h, x:x+pattern_w] += pattern_stacks.shape[0]

                if len(pattern_stack_list) > 1:
                    pattern_std = torch.std(pattern_stacks, dim=0)
                else:
                    pattern = img_t[y:y+pattern_h, x:x+pattern_w]
                    pattern_stack_list.append(pattern)
                    pattern_stacks = torch.stack(pattern_stack_list)
                    pattern_std = torch.std(pattern_stacks, dim=0)

                    if not is_border:
                        pattern_std /= 2.0

                sub_std = std_t[y:y+pattern_h, x:x+pattern_w]
                std_t[y:y+pattern_h, x:x+pattern_w] = torch.min(sub_std, pattern_std)
        else:
            if max_scores[i] == 0:
                std_t[ori_y:ori_y+pattern_h, ori_x:ori_x+pattern_w] = 255.0
            else:
                std_t[ori_y:ori_y+pattern_h, ori_x:ori_x+pattern_w] = 1.0

    weight_t = weight_t.clamp_min(1.0)
    mean_t = sum_t / weight_t
    diff_t = img_t - mean_t
    diff_t[diff_t < 0] *= -2

    std_t = torch.clamp_min(std_t/10, 1.0)
    amap = diff_t / std_t

    if kwargs.get('debug', False):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
        axes[0, 0].set_title('img')
        axes[0, 0].imshow(img_t.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title('mean')
        axes[1, 0].imshow(mean_t.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
        axes[2, 0].set_title('diff')
        axes[2, 0].imshow(diff_t.cpu().numpy(), cmap='gray', vmin=0, vmax=128)
        axes[0, 1].set_title('weights')
        axes[0, 1].imshow(weight_t.cpu().numpy(), cmap='gray', vmin=0, vmax=50)
        axes[1, 1].set_title('std')
        axes[1, 1].imshow(std_t.cpu().numpy())
        axes[2, 1].set_title('amap')
        axes[2, 1].imshow(amap.cpu().numpy(), cmap='gray', vmin=0, vmax=255)

    return amap


def get_saliency_map_pro(img_t,
                         pattern_shape_s=[(384, 8), (8, 384)],
                         stride_s=[(128, 8), (8, 128)], **kwargs):
    '''
    ori_img_t: (H,W)
    pattern_shape_s (list): list of pattern_shape
    stride_s (list): list of each stride
    '''
    import torch

    img_t = img_t.type(torch.float32)
    img_h, img_w = img_t.shape[:2]

    amap_t = torch.full_like(img_t, 255.0, dtype=torch.float32, device=img_t.device)
    for pattern_shape, stride in zip(pattern_shape_s, stride_s):
        pattern_h, pattern_w = pattern_shape
        stride_h, stride_w = stride

        x_strides = np.linspace(
            0, img_w-pattern_w, (img_w-pattern_w)//stride_w + 1, dtype=np.int)

        y_strides = np.linspace(
            0, img_h-pattern_h, (img_h-pattern_h)//stride_h + 1, dtype=np.int)

        # HxW -> (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x pattern_h x pattern_w
        img_unfold_t = img_t.unfold(0, pattern_h, 1).unfold(1, pattern_w, 1)

        # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h) x (pattern_w)
        # --->
        # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h * pattern_w)
        img_unfold_t = img_unfold_t.reshape([img_unfold_t.shape[0], img_unfold_t.shape[1], -1])

        unfold_h, unfold_w = img_unfold_t.shape[:2]
        pick_unfold_idx = [y*unfold_w+x for y in y_strides for x in x_strides]

        # (img_h - pattern_h + 1)x(img_w - pattern_w + 1) x (pattern_h * pattern_w)
        # --->
        # ((img_h - pattern_h + 1)*(img_w - pattern_w + 1)) x (pattern_h * pattern_w)
        img_unfold_t = img_unfold_t.reshape([img_unfold_t.shape[0]*img_unfold_t.shape[1], -1])

        pick_img_unfold_t = img_unfold_t[pick_unfold_idx]

        # get template match score
        # scores: (len(y_strides) * len(x_strides)) x (unfold_h * unfold_w)
        scores = ncc(pick_img_unfold_t, img_unfold_t)

        max_v = 255.0 * 255.0 * pattern_h * pattern_w

        for i, score in enumerate(scores):
            idx = pick_unfold_idx[i]
            ori_y = idx // unfold_w
            ori_x = idx % unfold_w
            score = score.reshape((unfold_h, unfold_w))
            #print(score.max())
            score[max(0, ori_y-pattern_h//2):min(ori_y+pattern_h//2, score.shape[0]),
                  max(0, ori_x-pattern_w//2):min(ori_x+pattern_w//2, score.shape[1])] = 0#max_v
            match_i = score.argmax()
            match_y, match_x = match_i // unfold_w, match_i % unfold_w
            ori_pattern = img_t[ori_y:ori_y+pattern_h, ori_x:ori_x+pattern_w]
            match_pattern = img_t[match_y:match_y+pattern_h, match_x:match_x+pattern_w]

            new_pattern_amap = torch.abs((ori_pattern - match_pattern))
            old_pattern_amap = amap_t[ori_y:ori_y+pattern_h, ori_x:ori_x+pattern_w]
            amap_t[ori_y:ori_y+pattern_h, ori_x:ori_x+pattern_w] = torch.min(new_pattern_amap, old_pattern_amap)
    return amap_t
