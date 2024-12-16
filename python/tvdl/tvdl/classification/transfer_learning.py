'''
Copyright (C) 2023 TuringVision

Base transfer learning classification model.
'''
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..utils.basic import freeze
from ..common import ClassificationModelBase

__all__ = ['mixup', 'TransferLearningModel']


class FreezeModelCb(Callback):
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        pl_module.freeze_to(pl_module.hparams.freeze_to_n)


def mixup(x, y_true, mixup_ratio, forward, loss):
    lambd = np.random.beta(mixup_ratio, mixup_ratio, x.size(0))
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lambd = x.new(lambd)
    # shuffle x, y
    shuffle = torch.randperm(x.size(0)).to(x.device)
    x1, y1 = x[shuffle], y_true[shuffle]
    out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
    # mixup x
    mixup_x = (x * lambd.view(out_shape) + x1 * (1 - lambd).view(out_shape))

    y_logits = forward(mixup_x)

    # mixup loss
    loss1, loss2 = loss(y_logits, y_true), loss(y_logits, y1)
    train_loss = (loss1 * lambd + loss2 * (1 - lambd)).mean()
    return train_loss


class TransferLearningModel(ClassificationModelBase):
    '''Transfer Learning with pre-trained models.


    Usage:
        train_dl, valid_dl = ill.dataloader(...)
        model = TransferLearningModel(len(ill.labelset()),
                                      backbone='resnet50')
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (torch.long):
            single label: (n,)
            multi label:  (n, num_classes)

    Model output:
        yp (torch.float32): (n, num_classes)
    '''

    def __init__(self,
                 num_classes: int,
                 backbone: str = "resnet18",
                 img_c: int = 3,
                 train_bn: bool = True,
                 lr: float = 1e-2,
                 multi_label: bool = False,
                 mixup_ratio: float = 0.0,
                 freeze_to_n: int = -1,
                 **kwargs,
                 ) -> None:
        """
        Args:
            num_classes (int):
            backbone (str): timm backbone
                use this code to show all supported backbones:

                import timm
                timm.list_models(pretrained=True)

            img_c (int): img channel 1 or 3, or other positive integer
                Number of input (color) channels.
                eg: 1-channel (gray image); 25-channel image (maybe satellite image)
            train_bn (bool): Whether the BatchNorm layers should be trainable
            lr (float): init learning rate
            multi_label (bool): Is multi label classification?
            mixup_ratio (float): 0 means no mixup
            freeze_to_n (int): layer index, -1 is last layer
        """
        super().__init__(**kwargs)
        # self.save_hyperparameters() not work after cython compile
        self.hparams.update({'num_classes': num_classes,
                             'backbone': backbone,
                             'img_c': img_c,
                             'train_bn': train_bn,
                             'lr': lr,
                             'multi_label': multi_label,
                             'mixup_ratio': mixup_ratio,
                             'freeze_to_n': freeze_to_n})
        self.build_model()
        self.valid_acc = pl.metrics.Accuracy(compute_on_step=False, threshold=1e-5)

    def build_model(self):
        '''
        '''
        from .backbones import TimmBackbone
        from .heads import create_base_head
        from ..utils.basic import num_features_model

        self.backbone = TimmBackbone(self.hparams.backbone, self.hparams.img_c, pretrained=self.hparams.pretrained)
        nf = num_features_model(self.backbone)
        self.head = create_base_head(nf, self.hparams.num_classes)

    def freeze_to(self, n=-1):
        freeze(module=self.backbone, n=n, train_bn=self.hparams.train_bn)
        self.hparams.freeze_to_n = n

    def configure_callbacks(self):
        return [FreezeModelCb()]

    def mixup(self, ratio=0.2):
        self.hparams.mixup_ratio = ratio

    def loss(self, logits, labels):
        if self.hparams.multi_label:
            with torch.no_grad():
                if labels.dim() == 1:
                    labels = F.one_hot(labels, num_classes=self.hparams.num_classes)
                labels = labels.type(torch.float32)
            return F.binary_cross_entropy_with_logits(input=logits, target=labels)
        else:
            return F.cross_entropy(input=logits, target=labels)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def predict(self, x):
        x = x.to(self.device)
        outputs = self.forward(x)
        return TransferLearningModel.post_process(outputs, self.hparams.multi_label)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        if self.hparams.mixup_ratio == 0:
            y_logits = self.forward(x)
            train_loss = self.loss(y_logits, y_true)
        else:
            train_loss = mixup(x, y_true, self.hparams.mixup_ratio, self.forward, self.loss)

        tqdm_dict = {"loss": train_loss}
        self.log_dict(tqdm_dict, prog_bar=True)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)

        val_loss = self.loss(y_pred, y_true)
        y_pred = F.softmax(y_pred, dim=-1)
        accuracy = self.valid_acc(y_pred.cpu(), y_true.cpu())

        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        valid_acc_mean = self.valid_acc.compute()
        self.valid_acc.reset()
        log_dict = {"val_loss": val_loss_mean, "val_acc": valid_acc_mean}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
