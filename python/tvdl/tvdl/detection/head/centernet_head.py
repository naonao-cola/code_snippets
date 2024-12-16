'''
Copyright (C) 2023 TuringVision

CenterNet Head .
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CenterNetHead']


class HeadConv(nn.Module):
    '''
    CenterNet HeadConve
    out_channels (int): head feature output channels
    in_channels (int): backbone output channels

    '''

    def __init__(self, out_channels: int, in_channels: int):
        super().__init__()
        self.out_channels = out_channels

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.fc(x)


class CenterNetHead(nn.Module):
    '''
    CenterNet Head
    in_channels (int): backbone output channels
    head_names (Tuple[str, ...])
    head_out_channels (Tuple[int, ...])
    loss_pos_balance (float): positive loss balance ratio
    '''

    def __init__(self, in_channels, head_names, head_out_channels, loss_pos_balance=0.5):
        super().__init__()

        self.loss_pos_balance = loss_pos_balance
        self.heads = {}
        for name, out_channel in zip(head_names, head_out_channels):
            self.heads[name] = out_channel
            self.__setattr__(name, HeadConv(out_channel, in_channels))
        self.init_weights()

    def forward(self, x):
        ret = {}
        for name in self.heads.keys():
            ret[name] = self.__getattr__(name)(x)
        return ret

    def init_weights(self):
        for name in self.heads.keys():
            if name.startswith("heatmap"):
                self.__getattr__(name).fc[-1].bias.data.fill_(-2.19)

    def neg_loss(self, pred, gt):
        fg_cnt = (gt > 0).sum()
        bg_cnt = (gt == 0).sum()
        pos_weight = bg_cnt * self.loss_pos_balance / max(fg_cnt, 2.0)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = BCEobj(pred, gt)
        return loss

    def reg_l1_loss(self, output, mask, ind, target):
        pred = self.transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def topk(self, scores, K=40):
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind // K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def heat_nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def gaussian_radius(self, det_size, min_overlap=0.3):
        height, width = det_size
        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = int(min(x, radius)), int(min(width - x, radius + 1))
        top, bottom = int(min(y, radius)), int(min(height - y, radius + 1))

        masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
        masked_gaussian = gaussian[
                          radius - top: radius + bottom, radius - left: radius + right
                          ]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def _gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y = torch.arange(-m, m + 1).unsqueeze(-1)
        x = torch.arange(-n, n + 1).unsqueeze(0)
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        feat = feat[:, ind[0]]
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
