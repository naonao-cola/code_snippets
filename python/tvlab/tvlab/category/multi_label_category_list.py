'''
Copyright (C) 2023 TuringVision

'''

from fastai.data_block import ItemList


class MultiLabelCategoryList(ItemList):
    _processor = list()
    def __init__(self, items, classes, **kwargs):
        assert isinstance(classes, (list, tuple))
        assert isinstance(classes[0], (list, tuple))
        super().__init__(items, **kwargs)
        self.classes = classes

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return tuple([self.classes[i].index(p) for i, p in enumerate(o)])

    def analyze_pred(self, pred, thresh=0.5):
        return [p.argmax() for p in pred]

    def reconstruct(self, t):
        raise NotImplementedError

    @property
    def c(self): return [len(cls) for cls in self.classes]
