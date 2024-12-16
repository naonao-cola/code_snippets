'''
Copyright (C) 2023 TuringVision

Find similar images from dataset
'''

import os
import os.path as osp
import pickle
import numpy as np
from tqdm.auto import tqdm, trange

__all__ = ['ImageSimilar', 'ImageSimilarPro']

def get_simialr_scores(train_actns, valid_actns, topk=5):
    import torch
    t = train_actns
    v = valid_actns
    w = t.norm(p=2, dim=1, keepdim=True)
    wv = v.norm(p=2, dim=1, keepdim=True)
    scores = torch.mm(v, t.t()) / (wv * w.t()).clamp(min=1e-8)
    score_sorted, _ = torch.sort(scores, dim=-1, descending=True)
    topk_score = score_sorted[:, :topk]
    total_scores = topk_score.sum(dim=1) / topk
    total_scores = total_scores.clamp(min=1e-8)
    return total_scores


def _largest_indices(arr, n):
    "Returns the `n` largest indices from a numpy array `arr`."
    #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    flat = arr.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, arr.shape)

class BasicImageSimilar:
    def __init__(self, img_path_list, similar_idxs, similar_scores):
        '''BasicImageSimilar: find similar images from dataset
        img_path_list: (list of str) N
        similar_idxs: (list of int list) (N, [0~k])
        similar_scores: (list of float list) (N, [0~k])
        '''
        assert len(similar_idxs) == len(similar_scores)

        self.img_path_list = img_path_list
        self._similar_idxs = similar_idxs
        self._similar_scores = similar_scores

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        '''
        index: (int) index in ill
            or (str) img_path
        return:
            idxs: (list of int) K * 2
            scores: (list of float) K
        '''
        if isinstance(index, str):
            img_path = index
            index = self.img_path_list.index(img_path)
        merged_idx = [index] + self._similar_idxs[index]
        merged_score = [1.0] + self._similar_scores[index]
        return merged_idx, merged_score

    def export(self, pkl_path):
        # save topk_index, topk_score
        os.makedirs(osp.dirname(pkl_path), exist_ok=True)
        pickle.dump(self, open(pkl_path, 'wb'))

    @classmethod
    def load(cls, pkl_path):
        # load topk_index, topk_score
        return pickle.load(open(pkl_path, 'rb'))

    def get(self, index, ill=None):
        '''
        index: (int) index in ill
            or (str) img_path
        ill: ImageLabelList

        return:
            idxs: (list of int) K * 2
            scores: (list of float) K
        '''
        if ill is None:
            return self[index]
        if isinstance(index, str):
            img_path = index
        else:
            img_path = ill.x[index]
        idxs, scores = self[img_path]
        new_idxs = []
        new_scores = []
        for idx, score in zip(idxs, scores):
            patha = self.img_path_list[idx]
            try:
                new_idxs.append(ill.x.index(patha))
                new_scores.append(score)
            except ValueError:
                pass
        return new_idxs, new_scores

    def group(self, ill=None):
        ''' group similar images
        In:
            ill: ImageLabelList
        Out:
            similar_idxs: (list of list) [[1,3,5], ...]
            similar_scores: (list of list) [[1.0, 0.9, 0.5], ...]
        '''
        if ill is None:
            return self._similar_idxs, self._similar_scores

        similar_idxs = []
        similar_scores = []
        for i in tqdm(range(len(self))):
            idxs, scores = self.get(i, ill)
            similar_idxs.append(idxs)
            similar_scores.append(scores)

        return similar_idxs, similar_scores

    def label_diff_group(self, ill):
        ''' group similar images (only group have different label)
            eg:
             group idxs:   [[1, 3, 5], ...]
             group labels: [['A', 'A', 'B'], ...]

        In:
            ill: ImageLabelList
        Out:
            similar_idxs: (list of list) [[1,3,5], ...]
            similar_scores: (list of list) [[1.0, 0.9, 0.5], ...]
        '''
        group_idxs, group_scores = self.group()
        diff_group_idxs = []
        diff_group_scores = []
        for idxs, scores in zip(group_idxs, group_scores):
            if len({ill.y[j] for j in idxs}) != 1:
                diff_group_idxs.append(idxs)
                diff_group_scores.append(scores)
        return diff_group_idxs, diff_group_scores

    def show(self, ill, index=None, diff=False, **kwargs):
        '''
        ill: ImageLabelList
        index: (int) index in ill
            or (str) img_path
        '''
        from ..ui import ImageCleaner
        if index is None:
            group_idxs, group_scores = self.label_diff_group(ill) if diff else self.group()
            idxs = [i for idx in group_idxs for i in idx]
            descs = [format(float(s), '.4f') for g in group_scores for s in g]
        else:
            idxs, scores = self.get(index, ill)
            descs = ['{:.4f}'.format(p) for p in scores]
            if isinstance(index, str):
                index = ill.x.index(index)
            if index != idxs[0]:
                idxs.insert(0, index)
                descs.insert(0, '{:.4f}'.format(1.0))
        return ImageCleaner(ill, idxs, descs, **kwargs)


class ImageSimilar(BasicImageSimilar):
    @classmethod
    def from_actns(cls, img_path_list, actns, topk=10, bs=512):
        '''
        img_path_list: (list of str)
        actns: (torch.Tensor) (N, W) images feature
        topk: (int)
        bs: (int)
        '''
        import torch

        t = actns.cuda()
        w = t.norm(p=2, dim=1, keepdim=True)
        total = t.shape[0]
        batch_n = int(np.ceil(total / bs))
        topk_index = torch.zeros(total, topk, dtype=torch.long)
        topk_score = torch.zeros(total, topk, dtype=torch.float32)
        # avoid out of gpu memory, split to batch
        for i in tqdm(range(batch_n)):
            si = i * bs
            se = min((i+1)*bs, total)
            tb = t[si:se]
            wb = w[si:se]
            ori_result = torch.mm(tb, t.t()) / (wb * w.t()).clamp(min=1e-8)
            for i, result_one in enumerate(ori_result):
                result_one[i+si] = 0
            ori_result_sorted, ori_indices = torch.sort(ori_result, dim=-1, descending=True)
            topk_index[si:se] = ori_indices[:, :topk]
            topk_score[si:se] = ori_result_sorted[:, :topk]

        return cls(img_path_list,
                   topk_index.numpy().astype(np.int32).tolist(),
                   topk_score.numpy().astype(np.float32).tolist())


class ImageORB:
    def __init__(self, n_cache=1000):
        '''ImageORB: Oriented FAST and Rotated BRIEF for accurate image similarity calculation
        n_cache: number of image desc cache
        '''
        import cv2
        self.orb = cv2.ORB_create()

        self._n_cache = n_cache
        self._cached_idxs = []
        self._cached_desc = []


    def _get_desc_from_cache(self, index):
        try:
            idx = self._cached_idxs.index(index)
            desc = self._cached_desc[idx]
            return desc
        except ValueError:
            return None

    def _cache_desc(self, index, desc):
        self._cached_idxs.append(index)
        self._cached_desc.append(desc)
        if len(self._cached_idxs) > self._n_cache:
            remove_cnt = self._n_cache // 10
            self._cached_idxs = self._cached_idxs[remove_cnt:]
            self._cached_desc = self._cached_desc[remove_cnt:]

    def _compute_desc(self, img):
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, desc = self.orb.detectAndCompute(img, None)
        return desc

    def get_desc(self, ill, index):
        ''' get the descriptors with ORB
        '''
        desc = self._get_desc_from_cache(index)
        if desc is None:
            img = ill[index][0]
            desc = self._compute_desc(img)
            self._cache_desc(index, desc)
        return desc

    def match_desc(self, desc1, desc2, ratio=0.6):
        ''' get similarity with two image's descriptors
        return:
            good_cnt
            percent
        '''
        try:
            import cv2
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(desc1, desc2, k=2)
            ratio = 0.75
            good_cnt = 0
            for m, n in matches:
                if m.distance < ratio*n.distance:
                    good_cnt += 1
            return good_cnt, good_cnt / max(len(desc1), len(desc2))
        except Exception as e:
            return 0, 0.0


class ImageSimilarPro(BasicImageSimilar):
    def __init__(self, img_path_list, similar_idxs, similar_scores):
        '''ImageSimilarPro: find high similarity images from dataset
        '''
        BasicImageSimilar.__init__(self, img_path_list, similar_idxs, similar_scores)
        self.repeat_idxs_group = None
        self.repeat_score_group = None

    @classmethod
    def from_basic_similar(cls, ill, similar, score_threshold=0.1, cnt_threshold=10):
        ''' create ImageSimilarPro from ImageSimilar
        In:
            ill: ImageLabelList
            similar: ImageSimilar
            score_threshold: threshold for orb match score
            cnt_threshold: threshold for orb matched keypoint count
        '''
        orb = ImageORB()

        similar_idxs = [None]*len(ill)
        similar_scores = [None]*len(ill)

        todo_idxs = list(range(len(ill)))
        prio_idxs = []
        done_idxs = []

        with trange(len(ill)) as t:
            while True:
                if prio_idxs:
                    idx = prio_idxs.pop(0)
                elif todo_idxs:
                    idx = todo_idxs.pop(0)
                else:
                    break
                topk_index, _ = similar[idx]
                done_idxs.append(idx)
                desc1 = orb.get_desc(ill, idx)
                one_idx = []
                one_score = []
                for i in topk_index:
                    is_similar = False
                    try:
                        score = similar_scores[i][similar_idxs[i].index(idx)]
                        is_similar = True
                    except Exception:
                        desc2 = orb.get_desc(ill, i)
                        good_cnt, score = orb.match_desc(desc1, desc2)
                        if score > score_threshold and good_cnt > cnt_threshold:
                            is_similar = True
                    if is_similar:
                        one_idx.append(i)
                        one_score.append(score)
                    if i in todo_idxs:
                        todo_idxs.remove(i)
                        prio_idxs.append(i)
                similar_idxs[idx] = one_idx
                similar_scores[idx] = one_score
                t.update()

        return cls(ill.x, similar_idxs, similar_scores)

    def _get_merged_result(self, idx, merged_idx=None, merged_score=None):
        idxs = self._similar_idxs[idx]
        scores = self._similar_scores[idx]
        if idxs is not None:
            for i, score in zip(idxs, scores):
                if i not in merged_idx:
                    if merged_idx is not None:
                        merged_idx.append(i)
                    else:
                        merged_idx = [i]
                    if merged_score is not None:
                        merged_score.append(score)
                    else:
                        merged_score = [score]
                    if len(merged_idx) < 1000:
                        merged_idx, merged_score = self._get_merged_result(i, merged_idx, merged_score)
        return merged_idx, merged_score

    def group(self):
        ''' group similar images
        In:
            ill: ImageLabelList
        Out:
            similar_idxs: (list of list) [[1,3,5], ...]
            similar_scores: (list of list) [[1.0, 0.9, 0.5], ...]
        '''
        if self.repeat_idxs_group and self.repeat_score_group:
            return self.repeat_idxs_group, self.repeat_score_group

        repeat_idxs = [i for i, idxs in enumerate(self._similar_idxs) if len(idxs) > 1]
        repeat_idxs_group = []
        repeat_score_group = []
        repeat_idxs_set = set()
        for idx in tqdm(repeat_idxs):
            group_idx, group_score = self[idx]
            group_idx_set = set(group_idx)
            if group_idx_set & repeat_idxs_set:
                if not group_idx_set.issubset(repeat_idxs_set):
                    for i in range(len(repeat_idxs_group)):
                        group_idx_ = repeat_idxs_group[i]
                        group_score_ = repeat_score_group[i]
                        if group_idx_set & set(group_idx_):
                            for idx, s in zip(group_idx, group_score):
                                if idx not in group_idx_:
                                    group_idx_.append(idx)
                                    group_score_.append(s)
                        repeat_idxs_group[i] = group_idx_
                        repeat_score_group[i] = group_score_
            else:
                repeat_idxs_group.append(group_idx)
                repeat_score_group.append(group_score)
            repeat_idxs_set.update(group_idx)
        self.repeat_idxs_group = repeat_idxs_group
        self.repeat_score_group = repeat_score_group
        return repeat_idxs_group, repeat_score_group

    def __getitem__(self, index):
        '''
        index: (int) index in ill
            or (str) img_path
        return:
            idxs: (list of int) K * 2
            scores: (list of float) K
        '''
        if isinstance(index, str):
            img_path = index
            index = self.img_path_list.index(img_path)
        merged_idx = [index]
        merged_score = [1.0]
        return self._get_merged_result(index, merged_idx, merged_score)
