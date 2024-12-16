'''
Copyright (C) 2023 TuringVision

Implements SHOT:
[ICML-2020] Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation
'''
import numpy as np
import torch
import torch.nn.functional as F
from ..transfer_learning import TransferLearningModel
from ...utils.basic import freeze

__all__ = ['SHOTModel']


class SHOTModel(TransferLearningModel):
    '''
    Implements SHOT:
    [ICML-2020] Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

    Usage:
        src_model_path = './path/of/model.ckpt'
        src_model = TransferLearningModel.load_from_checkpoint(src_model_path)
        model = SHOT.from_src(src_model, lr=0.01)
        trainer.fit(model, target_train_dl, target_valid_dl)
    '''
    @classmethod
    def from_src(cls,
                 src_model: TransferLearningModel,
                 lr: float = 1e-2,
                 loss_ent_w: float = 1.0,
                 loss_cls_w: float = 0.3,
                 cnt_threshold: int = 0,
                 **kwargs):
        src_model.__class__ = SHOTModel
        self = src_model
        self.hparams.lr = lr
        self.hparams.freeze_to_n = 0
        self.loss_ent_w = loss_ent_w
        self.loss_cls_w = loss_cls_w
        self.cnt_threshold = cnt_threshold
        return self

    def freeze_to(self, n=0):
        freeze(self.backbone, n, True)
        freeze(self.head, -1, False)

    def loss(self, logits, labels):
        softmax_out = F.softmax(logits, dim=1)
        entropy = -softmax_out * torch.log(softmax_out + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        ent_loss = torch.mean(entropy)
        cls_loss = super().loss(logits, labels)
        return self.loss_ent_w * ent_loss + self.loss_cls_w * cls_loss

    def get_pseudo_label(self, loader):
        from scipy.spatial.distance import cdist
        from fastprogress.fastprogress import progress_bar as pbar
        all_f = []
        all_out = []
        with torch.no_grad():
            for x, _ in pbar(loader, leave=False):
                x = x.to(self.device)
                feature = self.backbone(x)
                out = self.head(feature)
                all_f.append(feature.float().cpu())
                all_out.append(out.float().cpu())
        all_f = torch.cat(all_f, 0)
        all_out = torch.cat(all_out, 0)
        all_out = F.softmax(all_out, dim=1)
        # origin model predict label
        _, predict = torch.max(all_out, 1)

        all_f = torch.cat((all_f, torch.ones(all_f.size(0), 1, dtype=torch.float32)), 1)
        all_f = (all_f.t() / torch.norm(all_f, p=2, dim=1)).t()
        all_f = all_f.float().cpu().numpy()
        K = all_out.size(1)
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > self.cnt_threshold)[0]

        aff = all_out.float().cpu().numpy()
        pred_label = predict
        for i in range(2):
            if i > 0:
                aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_f)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_f, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]
        return pred_label.astype('int')

    def on_train_epoch_start(self):
        ''' update pseudo label when epoch start
        '''
        from torch.utils.data import SequentialSampler, DataLoader

        self.zero_grad()
        torch.set_grad_enabled(False)

        loader = self.train_dataloader()
        dset = loader.dataset.copy()
        dset.set_tfms(self.val_dataloader().dataset.get_tfms())
        seq_loader = DataLoader(dset,
                                batch_size=loader.batch_size,
                                shuffle=False,
                                num_workers=loader.num_workers,
                                pin_memory=loader.pin_memory,
                                collate_fn=loader.collate_fn,
                                drop_last=False)

        loader.dataset.y = self.get_pseudo_label(seq_loader)
        torch.set_grad_enabled(True)
