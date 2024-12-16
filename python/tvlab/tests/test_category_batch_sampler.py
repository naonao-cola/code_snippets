import unittest
from tvlab.category import batch_sampler
import torch
from tvlab import ImageLabelList
import os.path as osp


def label_fun(x):
    return torch.tensor([1]) if 'bees' in x else torch.tensor([0])


class TestBatchSampler(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBatchSampler, self).__init__(*args, **kwargs)
        self.data_path = osp.normpath('data/hymenoptera_data/')
        self.bill = ImageLabelList.from_folder(self.data_path).label_from_folder()
        self.bill = self.bill.label_from_func(label_fun)
        self.samper = torch.utils.data.RandomSampler(self.bill)

    def test_OriginBatchSampler(self):
        ba_sample = batch_sampler.OriginBatchSampler(self.samper, 5, 2)
        self.assertEqual(len(ba_sample), 2)

    def test_PairBatchSampler(self):
        ba_sample = batch_sampler.PairBatchSampler(self.samper, 5, 2)
        self.assertEqual(len(ba_sample), 1)

    def test_BalanceBatchSampler(self):
        ba_sample = batch_sampler.BalanceBatchSampler(self.samper, 2)
        self.assertEqual(len(ba_sample), 5)

    def test_ReverseBatchSampler(self):
        ba_sample = batch_sampler.ReverseBatchSampler(self.samper, 2)
        self.assertEqual(len(ba_sample), 5)

    def test_MultiBatchSampler(self):
        ba_sample = batch_sampler.MultiBatchSampler(self.samper, 2, sampler_type=('origin', 'reverse'))
        self.assertEqual(len(ba_sample), 5)


if __name__ == '__main__':
    unittest.main()
