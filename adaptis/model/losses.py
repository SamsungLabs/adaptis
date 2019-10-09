import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptis.model import metrics
from adaptis.utils import misc


class NormalizedFocalLossSoftmax(nn.Module):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, detach_delimeter=True, gamma=2, eps=1e-10):
        super(NormalizedFocalLossSoftmax, self).__init__()
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._eps = eps
        self._gamma = gamma
        self._k_sum = 0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.unsqueeze(1)
        softmaxout = F.softmax(pred, dim=1)

        t = label != self._ignore_label
        pt = torch.gather(softmaxout, index=label.long(), dim=1)
        pt = torch.where(t, pt, torch.ones_like(pt))
        beta = (1 - pt) ** self._gamma

        t_sum = torch.sum(t, dim=(-2, -1), keepdim=True).float()
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True).float()
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.cpu().numpy().mean()

        loss = -beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))

        if self._size_average:
            bsum = torch.sum(t_sum, dim=misc.get_dims_with_exclusion(t_sum.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0
        t = torch.ones_like(one_hot)

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        pt = torch.where(one_hot, pred, 1 - pred)
        pt = torch.where(label != self._ignore_label, pt, torch.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        t_sum = torch.sum(t, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult

        ignore_area = torch.sum(label == -1, dim=tuple(range(1, label.dim()))).cpu().numpy()
        sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        sample_weight = label != self._ignore_label

        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = torch.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        sample_weight = label != -1

        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(label == 1, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class AdaptISProposalsLossIoU(nn.Module):
    def __init__(self, batch_size, bad_percentile=30, init_iou_thresh=0.35,
                 from_logits=True, ignore_label=-1):
        super(AdaptISProposalsLossIoU, self).__init__()
        self._batch_size = batch_size
        self._bad_percentile = bad_percentile
        self._from_logits = from_logits
        self._iou_metric = metrics.AdaptiveIoU(init_thresh=init_iou_thresh,
                                               from_logits=False, ignore_label=ignore_label)
        self._ignore_label = ignore_label
        self._ce_loss = SigmoidBinaryCrossEntropyLoss()

    def forward(self, pred_inst_maps, pred_proposals, gt_inst_masks):
        with torch.no_grad():
            if self._from_logits:
                pred_inst_maps = torch.sigmoid(pred_inst_maps)

            self._iou_metric.update(pred_inst_maps, gt_inst_masks)

            pred_masks = pred_inst_maps > self._iou_metric.iou_thresh
            batch_iou = metrics._compute_iou(pred_masks, gt_inst_masks > 0, gt_inst_masks == self._ignore_label,
                                             keep_ignore=True)

            batch_iou = batch_iou.reshape((self._batch_size, -1))

            prob_score = torch.sum(pred_inst_maps * gt_inst_masks,
                                   dim=misc.get_dims_with_exclusion(gt_inst_masks.dim(), 0))

            prob_score = prob_score / torch.max(
                torch.sum(gt_inst_masks, dim=misc.get_dims_with_exclusion(gt_inst_masks.dim(), 0)),
                torch.ones(1, dtype=torch.float).to(gt_inst_masks.device)
            )

            prob_score = prob_score.cpu().numpy().reshape((self._batch_size, -1))

        labels = []
        for i in range(self._batch_size):
            obj_iou = batch_iou[i]
            if obj_iou.min() < 0 or obj_iou.max() < 1e-5:
                labels.append([-1] * len(obj_iou))
                continue

            obj_score = obj_iou * prob_score[i]
            if obj_iou.max() - obj_iou.min() > 0.1:
                th = np.percentile(obj_score, self._bad_percentile) * 0.95
            else:
                th = 0.99 * obj_score.min()

            good_points = obj_score > th
            gp_min_score = obj_score[good_points].min()
            gp_max_score = obj_score[good_points].max()

            obj_labels = (obj_score > th).astype(np.float32)
            if gp_max_score - gp_min_score > 1e-3:
                prob = (obj_score[good_points] - gp_min_score) / (gp_max_score - gp_min_score)
                obj_labels[good_points] = 0.7 + 0.3 * prob
            labels.append(obj_labels.tolist())

        labels = np.array(labels)
        labels = torch.from_numpy(labels).to(pred_proposals.device)

        return self._ce_loss(pred_proposals, labels)

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_iou_thresh', value=self._iou_metric.iou_thresh, global_step=global_step)
