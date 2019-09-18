import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like
from adaptis.model.metrics import AdaptiveIoU, _compute_iou


class NormalizedFocalLossSoftmax(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, detach_delimeter=True, gamma=2, eps=1e-10, **kwargs):
        super(NormalizedFocalLossSoftmax, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._eps = eps
        self._gamma = gamma
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label):
        label = F.expand_dims(label, axis=1)
        softmaxout = F.softmax(pred, axis=1)

        t = label != self._ignore_label
        pt = F.pick(softmaxout, label, axis=1, keepdims=True)
        pt = F.where(t, pt, F.ones_like(pt))
        beta = (1 - pt) ** self._gamma

        t_sum = F.cast(F.sum(t, axis=(-2, -1), keepdims=True), 'float32')
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -beta * F.log(F.minimum(pt + self._eps, 1))

        if self._size_average:
            bsum = F.sum(t_sum, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class NormalizedFocalLossSigmoid(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1, **kwargs):
        super(NormalizedFocalLossSigmoid, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        one_hot = label > 0
        t = F.ones_like(one_hot)

        if not self._from_logits:
            pred = F.sigmoid(pred)

        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        pt = F.where(one_hot, pred, 1 - pred)
        pt = F.where(label != self._ignore_label, pt, F.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)

        ignore_area = F.sum(label == -1, axis=0, exclude=True).asnumpy()
        sample_mult = F.mean(mult, axis=0, exclude=True).asnumpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != self._ignore_label

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            bsum = F.sum(sample_weight, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(label == 1, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, grad_scale=1.0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._grad_scale = grad_scale

    def hybrid_forward(self, F, pred, label):
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            grad_scale=self._grad_scale,
        )
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SigmoidBinaryCrossEntropyLoss(Loss):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label):
        label = _reshape_like(F, label, pred)
        sample_weight = label != self._ignore_label
        label = F.where(sample_weight, label, F.zeros_like(label))

        if not self._from_sigmoid:
            loss = F.relu(pred) - pred * label + \
                F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            eps = 1e-12
            loss = -(F.log(pred + eps) * label
                     + F.log(1. - pred + eps) * (1. - label))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class AdaptISProposalsLossIoU(gluon.HybridBlock):
    def __init__(self, batch_size, bad_percentile=30, init_iou_thresh=0.35,
                 from_logits=True, ignore_label=-1):
        super(AdaptISProposalsLossIoU, self).__init__()
        self._batch_size = batch_size
        self._bad_percentile = bad_percentile
        self._from_logits = from_logits
        self._iou_metric = AdaptiveIoU(init_thresh=init_iou_thresh,
                                       from_logits=False, ignore_label=ignore_label)
        self._ignore_label = ignore_label
        self._ce_loss = SigmoidBinaryCrossEntropyLoss()

    def hybrid_forward(self, F, pred_inst_maps, pred_proposals, gt_inst_masks):
        with mx.autograd.pause():
            if self._from_logits:
                pred_inst_maps = mx.nd.sigmoid(pred_inst_maps)

            self._iou_metric.update(pred_inst_maps, gt_inst_masks)
            pred_masks = pred_inst_maps > self._iou_metric.iou_thresh

            batch_iou = _compute_iou(pred_masks, gt_inst_masks > 0, gt_inst_masks == self._ignore_label,
                                     keep_ignore=True)
            batch_iou = batch_iou.reshape((self._batch_size, -1))

            prob_score = mx.nd.sum(pred_inst_maps * gt_inst_masks, axis=0, exclude=True)
            prob_score = prob_score / mx.nd.maximum(mx.nd.sum(gt_inst_masks, axis=0, exclude=True), 1)
            prob_score = prob_score.asnumpy().reshape((self._batch_size, -1))

        labels = []
        for i in range(self._batch_size):
            obj_iou = batch_iou[i]
            if obj_iou.min() < 0:
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
        labels = nd.array(labels).as_in_context(pred_proposals.context)

        return self._ce_loss(pred_proposals, labels)

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_iou_thresh', value=self._iou_metric.iou_thresh, global_step=global_step)
