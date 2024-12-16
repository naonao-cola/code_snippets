'''
Copyright (C) 2023 TuringVision

BatchSampler for category task.
'''
from random import shuffle
from torch.utils.data import BatchSampler

__all__ = ['UniformChoicer', 'PairBatchSampler', 'MultiBatchSampler']


class UniformChoicer:
    def __init__(self, items):
        self.items = items
        self._remain_items = list()

    def __call__(self, k=1):
        import copy
        k_items = []
        while k > 0:
            if len(self._remain_items) <= k:
                items = self.items.copy()
                shuffle(items)
                self._remain_items += items
            k_items += self._remain_items[:k]
            self._remain_items = self._remain_items[k:]
            k -= len(self.items)
        return k_items


class PairBatchSampler(BatchSampler):
    def __init__(self, sampler, n_img, n_cls, get_y_func=None):
        '''PairBatchSampler: Always sample n_cls in a batch,
        for each class, we randomly sample n_img images.

        n_img: number of images for each class
        n_cls: number of class
        '''
        n_img = int(n_img)
        n_cls = int(n_cls)
        self.sampler = sampler
        self.batch_size = n_img * n_cls
        n_cls_idxs = {}
        for idx in self.sampler:
            if not get_y_func:
                y = self.sampler.data_source.y[idx].data
            else:
                y = get_y_func(self.sampler.data_source.y[idx])
            if y not in n_cls_idxs:
                n_cls_idxs[y] = [idx]
            else:
                n_cls_idxs[y].append(idx)
        self.n_cls = n_cls
        self.n_img = n_img
        self.n_cls_idxs = n_cls_idxs
        self.n_cls_choicer = {k:UniformChoicer(v) for k, v in self.n_cls_idxs.items()}

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        import copy
        cls_choicer = UniformChoicer(list(self.n_cls_idxs.keys()))
        for _ in range(len(self)):
            pick_cls = cls_choicer(self.n_cls)
            batch = []
            for c in pick_cls:
                idxs = self.n_cls_choicer[c](self.n_img)
                batch += idxs
            yield batch


class OriginBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, get_y_func=None):
        '''OriginBatchSampler:
        sampler: Base sampler
        batch_size: total size of mini-batch
        '''
        self.sampler = sampler
        self.batch_size = batch_size
        self.resample_idxs = [idx for idx in self.sampler]

    def __len__(self):
        return len(self.resample_idxs) // self.batch_size

    def __iter__(self):
        shuffle(self.resample_idxs)
        batch = []
        for idx in self.resample_idxs:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


class BalanceBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, get_y_func=None):
        '''BalanceBatchSampler:
        sampler: Base sampler
        batch_size: total size of mini-batch
        '''
        self.sampler = sampler
        self.batch_size = batch_size

        n_cls_idxs = {}
        for idx in self.sampler:
            if not get_y_func:
                y = self.sampler.data_source.y[idx].data
            else:
                y = get_y_func(self.sampler.data_source.y[idx])
            if y not in n_cls_idxs:
                n_cls_idxs[y] = [idx]
            else:
                n_cls_idxs[y].append(idx)
        max_imgs = max([len(idxs) for idxs in n_cls_idxs.values()])
        resample_idxs = []
        for idxs in n_cls_idxs.values():
            choicer = UniformChoicer(idxs)
            resample_idxs += choicer(max_imgs)

        self.resample_idxs = resample_idxs

    def __len__(self):
        return len(self.resample_idxs) // self.batch_size

    def __iter__(self):
        shuffle(self.resample_idxs)
        batch = []
        for idx in self.resample_idxs:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


class ReverseBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, get_y_func=None):
        '''ReverseBatchSampler:
        sampler: Base sampler
        batch_size: total size of mini-batch
        '''
        self.sampler = sampler
        self.batch_size = batch_size

        n_cls_idxs = {}
        for idx in self.sampler:
            if not get_y_func:
                y = self.sampler.data_source.y[idx].data
            else:
                y = get_y_func(self.sampler.data_source.y[idx])
            if y not in n_cls_idxs:
                n_cls_idxs[y] = [idx]
            else:
                n_cls_idxs[y].append(idx)

        cls_cnt_list = [(c, len(idxs)) for c, idxs in n_cls_idxs.items()]
        cls_cnt_list = sorted(cls_cnt_list, key=lambda x:x[1])
        cls_list, cnt_list = list(zip(*cls_cnt_list))
        cnt_list = cnt_list[::-1]

        self.n_cls_choicer = {k:UniformChoicer(v) for k, v in n_cls_idxs.items()}

        resample_cls = []
        for cls, cnt in zip(cls_list, cnt_list):
            resample_cls += [cls] * cnt
        self.cls_choicer = UniformChoicer(resample_cls)

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        for _ in range(len(self)):
            pick_cls = self.cls_choicer(self.batch_size)
            batch = []
            for c in pick_cls:
                idxs = self.n_cls_choicer[c](1)
                batch += idxs
            yield batch


class MultiBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, sampler_type=('origin', 'reverse'), get_y_func=None):
        '''MultiBatchSampler: Multi different sampler are used in one batch, and each sampler selects batch_size//N data.
        sampler: Base sampler
        batch_size: total size of mini-batch
        sampler_type (tuple): (type_a, type_b, ...) , type is one of ['origin', 'balance', 'reverse']
        '''
        self.sampler = sampler
        self.batch_size = batch_size
        type_cls_map = {'origin': OriginBatchSampler,
                        'balance': BalanceBatchSampler,
                        'reverse': ReverseBatchSampler}
        N_sampler = len(sampler_type)
        one_sampler_bs = int(batch_size/N_sampler)
        self.sampler_list = [type_cls_map[type_i](sampler, one_sampler_bs, get_y_func=get_y_func)
                             for type_i in sampler_type]

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        for sa_batch in zip(*self.sampler_list):
            batch = [i for idxs in sa_batch for i in idxs]
            yield batch
