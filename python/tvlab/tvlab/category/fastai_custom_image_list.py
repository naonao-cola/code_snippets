'''
Copyright (C) 2023 TuringVision

fasti custom ImageList.
'''
from fastai.vision import ImageList


class CImageList(ImageList):
    def __init__(self, ill):
        super().__init__(ill.x)
        self.ill = ill

    def get(self, i):
        return self.ill[i][0]
