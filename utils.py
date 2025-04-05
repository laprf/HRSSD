import os
import random

import numpy as np
import torch

cuda = torch.cuda.is_available()


def _S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg
    return Q


def _object(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

    return score


def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    # print(Q)
    return Q


def _centroid(gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        if cuda:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if cuda:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return X.long(), Y.long()


def _divideGT(gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4


def _dividePrediction(pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB


def _ssim(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    random.seed(seed)  # 可以注释掉
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 这个懂吧
    torch.backends.cudnn.deterministic = True  # 确定性固定
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def mean_square_error(pred, gt):
    """Compute mean square error between prediction and ground truth."""
    return torch.mean(torch.abs(gt - pred))


def eval_smeasure(pred, gt):
    """Evaluate S-measure metric."""
    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    with torch.no_grad():
        for i in range(len(pred)):
            pred_ = pred[i]
            gt_ = gt[i]
            y = gt_.mean()
            if y == 0:
                x = pred_.mean()
                q = 1.0 - x
            elif y == 1:
                x = pred_.mean()
                q = x
            else:
                gt_[gt_ >= 0.5] = 1
                gt_[gt_ < 0.5] = 0
                q = alpha * _S_object(pred_, gt_) + (1 - alpha) * _S_region(pred_, gt_)
                if q.item() < 0:
                    q = torch.FloatTensor([0.0])
            img_num += 1.0
            avg_q += q.item()
        avg_q /= img_num
    return avg_q
