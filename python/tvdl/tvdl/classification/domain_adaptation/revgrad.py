'''
Copyright (C) 2023 TuringVision

Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Function
from pytorch_lightning.callbacks import Callback
from ...utils.basic import freeze

__all__ = ['RevGradModel']


class FreezeModelCb(Callback):
    def on_before_accelerator_backend_setup(self, trainer, pl_module):
        pl_module.freeze_to(pl_module.hparams.freeze_to_n)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class RevGradModel(pl.LightningModule):
    '''
    Implements RevGrad:
    Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
    Domain-adversarial training of neural networks, Ganin et al. (2016)

    Usage:
        You must provide two train_dataLoader: [source, tareget]

        s_train_dl, s_valid_dl = ill_source.dataloader(...)
        t_train_dl, t_valid_dl = ill_target.dataloader(...)
        trainer.fit(model, [s_train_dl, t_train_dl], t_valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (torch.long):(n,)

    Model output:
        yp (torch.float32): (n, num_classes)
    '''
    def __init__(
            self,
            num_classes: int,
            backbone: str = "resnet18",
            img_c: int = 3,
            train_bn: bool = True,
            lr: float = 1e-2,
            freeze_to_n: int = 0,
            grad_lambda: float = 1.0,
            loss_domain_w: float = 1.0,
            loss_cls_w: float = 1.0,
            loss_ent_w: float = 0.0,
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
            freeze_to_n (int): layer index, -1 is last layer
            grad_lambda (float): reversal gradident weight
            loss_domain_w (float): domain loss weight
            loss_cls_w (float): classification loss weight
            loss_ent_w (float): target entroy loss weight
        """
        super().__init__()
        self.hparams.update({'num_classes': num_classes,
                             'backbone': backbone,
                             'img_c': img_c,
                             'train_bn': train_bn,
                             'lr': lr,
                             'freeze_to_n': freeze_to_n,
                             'grad_lambda': grad_lambda,
                             'loss_domain_w': loss_domain_w,
                             'loss_cls_w': loss_cls_w,
                             'loss_ent_w': loss_ent_w})
        self.build_model()
        self.valid_acc = pl.metrics.Accuracy(compute_on_step=False, threshold=1e-5)

    def build_model(self):
        '''
        '''
        from ..backbones import TimmBackbone
        from ..heads import create_base_head
        from ...utils.basic import num_features_model

        self.backbone = TimmBackbone(self.hparams.backbone, self.hparams.img_c)
        nf = num_features_model(self.backbone)
        self.head = create_base_head(nf, self.hparams.num_classes)
        self.discriminator = nn.Sequential(GradientReversal(self.hparams.grad_lambda),
                                           create_base_head(nf, 1))

    def configure_callbacks(self):
        return [FreezeModelCb()]

    def freeze_to(self, n=-1):
        freeze(module=self.backbone, n=n, train_bn=self.hparams.train_bn)
        self.hparams.freeze_to_n = n

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def predict(self, x):
        x = x.to(self.device)
        y_pred = self.forward(x)
        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred

    def loss(self, logits, labels):
        return F.cross_entropy(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        source_batch, target_batch = batch
        sx, sy = source_batch
        tx, _ = target_batch

        x = torch.cat([sx, tx])
        domain_y = torch.cat(
            [torch.ones(sx.shape[0], dtype=torch.float32),
             torch.zeros(tx.shape[0], dtype=torch.float32)])
        domain_y = domain_y.to(self.device)

        features = self.backbone(x)
        domain_preds = self.discriminator(features).squeeze()
        label_preds = self.head(features)

        loss_domain = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
        loss_domain = self.hparams.loss_domain_w * loss_domain

        loss_cls = self.loss(label_preds[:sx.shape[0]], sy)
        loss_cls = self.hparams.loss_cls_w * loss_cls

        softmax_out = F.softmax(label_preds[sx.shape[0]:], dim=1)
        entropy = -softmax_out * torch.log(softmax_out + 1e-5)
        loss_ent = torch.mean(torch.sum(entropy, dim=1))
        loss_ent = self.hparams.loss_ent_w * loss_ent

        train_loss = loss_domain + loss_cls + loss_ent
        tqdm_dict = {'loss_domain': loss_domain, "loss_cls": loss_cls, "loss_ent": loss_ent}
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
        val_loss_mean = torch.stack([output["val_loss"]
                                     for output in outputs]).mean()
        valid_acc_mean = self.valid_acc.compute()
        self.valid_acc.reset()
        log_dict = {"val_loss": val_loss_mean, "val_acc": valid_acc_mean}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]
