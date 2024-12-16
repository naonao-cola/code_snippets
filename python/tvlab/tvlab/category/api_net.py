'''
Copyright (C) 2023 TuringVision

training forward and loss_func for api-net.
see "Learning Attentive Pairwise Interaction for Fine-Grained Classification"
'''
__all__ = ['api_net_training_forward', 'api_net_loss']


def api_net_training_forward(x, batch_n_cls):
    import torch
    import torch.nn.functional as F
    def _get_pair_idxs(s, row, col, n_imgs):
        '''
        return: [[1,3], [2,4]]
        '''
        pair_idxs = []
        b = s[row*n_imgs:(row+1)*n_imgs, col*n_imgs:(col+1)*n_imgs]
        for y in range(b.shape[0]):
            x = b[y].argmax().item()
            x += col * n_imgs
            y += row * n_imgs
            pair_idxs.append([x, y])
        return pair_idxs

    # 1. pair idxs construction
    with torch.no_grad():
        w = x.norm(p=2, dim=1, keepdim=True)
        scores = torch.mm(x, x.t()) / (w * w.t()).clamp(min=1e-8)
        scores += 1.0
        scores = scores.masked_fill(torch.eye(scores.shape[0],
                                              dtype=torch.bool,
                                              device=scores.device), 0)
        pair_idxs = []
        n_imgs = int(scores.shape[0] // batch_n_cls)
        for row in range(batch_n_cls):
            for col in range(row, batch_n_cls):
                pair_idxs += _get_pair_idxs(scores, row, col, n_imgs)
        x1_idxs, x2_idxs = list(zip(*pair_idxs))
        x1_idxs = list(x1_idxs)
        x2_idxs = list(x2_idxs)
    # 2. get x1, x2
    x1 = x[x1_idxs]
    x2 = x[x2_idxs]

    # 3. get xm
    xm = x1 + x2

    # 4. get g1, g2
    g1 = torch.sigmoid(xm * x1)
    g2 = torch.sigmoid(xm * x2)

    # 5. get x1self, x2self, x1other, x2other
    x1self = x1 + x1 * g1
    x2self = x2 + x2 * g2
    x1other = x1 + x1 * g2
    x2other = x2 + x2 * g1

    return (x1self, x2self, x1other, x2other, x1_idxs, x2_idxs)


def api_net_loss(predict, target, ce_loss_func, loss_rk_coeff=1.0, loss_p_margin=0.05):
    import torch
    import torch.nn.functional as F

    if isinstance(predict, tuple):
        h1self, h2self, h1other, h2other, x1_idxs, x2_idxs = predict

        x1_target = target[x1_idxs]
        loss_ce_h1_self = ce_loss_func(h1self, x1_target)
        loss_ce_h1_other = ce_loss_func(h1other, x1_target)

        x2_target = target[x2_idxs]
        loss_ce_h2_self = ce_loss_func(h2self, x2_target)
        loss_ce_h2_other = ce_loss_func(h2other, x2_target)
        loss_ce = loss_ce_h1_self + loss_ce_h1_other + loss_ce_h2_self + loss_ce_h2_other

        p1self = F.softmax(h1self, dim=0)
        p2self = F.softmax(h2self, dim=0)

        p1other = F.softmax(h1other, dim=0)
        p2other = F.softmax(h2other, dim=0)

        p1self_c = p1self[torch.arange(p1self.shape[0]), x1_target]
        p1other_c = p1other[torch.arange(p1other.shape[0]), x1_target]
        p2self_c = p2self[torch.arange(p2self.shape[0]), x2_target]
        p2other_c = p2other[torch.arange(p2other.shape[0]), x2_target]

        loss_rk1 = p1other_c - p1self_c + loss_p_margin
        loss_rk1 = torch.clamp_min(loss_rk1, 0.0)
        loss_rk1 = loss_rk1.mean()
        loss_rk2 = p2other_c - p2self_c + loss_p_margin
        loss_rk2 = torch.clamp_min(loss_rk2, 0.0)
        loss_rk2 = loss_rk2.mean()
        loss_rk = loss_rk1 + loss_rk2

        loss = loss_ce + loss_rk_coeff * loss_rk
    else:
        loss = ce_loss_func(predict, target)
    return loss
