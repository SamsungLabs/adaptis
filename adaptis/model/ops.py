import mxnet as mx
from mxnet import gluon


class AdaIN(gluon.HybridBlock):
    def __init__(self, channels, norm=True):
        super(AdaIN, self).__init__()
        self.channels = channels
        self.norm = norm

        with self.name_scope():
            self.affine_scale = gluon.nn.Dense(channels,
                                               bias_initializer=mx.init.Constant(1),
                                               use_bias=True, flatten=True)
            self.affine_bias = gluon.nn.Dense(channels,
                                             use_bias=True, flatten=True)

    def hybrid_forward(self, F, x, w):
        ys = self.affine_scale(w)
        yb = self.affine_bias(w)
        ys = F.expand_dims(ys, axis=2)
        yb = F.expand_dims(yb, axis=2)

        xm = F.reshape(x, shape=(0, 0, -1))  # (N,C,H,W) --> (N,C,K)
        if self.norm:
            xm_mean = F.mean(xm, axis=2, keepdims=True)
            xm_centered = F.broadcast_minus(xm, xm_mean)
            xm_std_rev = F.rsqrt(F.mean(F.square(xm_centered), axis=2, keepdims=True))  # 1 / std (rsqrt, not sqrt)
            xm_norm = F.broadcast_mul(xm_centered, xm_std_rev)
        else:
            xm_norm = xm

        xm_scaled = F.broadcast_plus(F.broadcast_mul(xm_norm, ys), yb)
        return F.reshape_like(xm_scaled, x)


class AppendCoordFeatures(gluon.HybridBlock):
    def __init__(self, norm_radius, append_dist=True, spatial_scale=1.0):
        super(AppendCoordFeatures, self).__init__()
        self.xs = None
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.append_dist = append_dist

    def _ctx_kwarg(self, x):
        if isinstance(x, mx.nd.NDArray):
            return {"ctx": x.context}
        return {}

    def get_coord_features(self, F, points, rows, cols, batch_size,
                           **ctx_kwarg):
        row_array = F.arange(start=0, stop=rows, step=1, **ctx_kwarg)
        col_array = F.arange(start=0, stop=cols, step=1, **ctx_kwarg)
        coord_rows = F.repeat(F.reshape(row_array, (1, 1, rows, 1)), repeats=cols, axis=3)
        coord_cols = F.repeat(F.reshape(col_array, (1, 1, 1, cols)), repeats=rows, axis=2)

        coord_rows = F.repeat(coord_rows, repeats=batch_size, axis=0)
        coord_cols = F.repeat(coord_cols, repeats=batch_size, axis=0)

        coords = F.concat(coord_rows, coord_cols, dim=1)

        add_xy = F.reshape(points * self.spatial_scale, shape=(0, 0, 1))
        add_xy = F.reshape(F.repeat(add_xy, rows * cols, axis=2),
                           shape=(0, 0, rows, cols))

        coords = (coords - add_xy) / (self.norm_radius * self.spatial_scale)
        if self.append_dist:
            dist = F.sqrt(F.sum(F.square(coords), axis=1, keepdims=1))
            coord_features = F.concat(coords, dist, dim=1)
        else:
            coord_features = coords

        coord_features = F.clip(coord_features, a_min=-1, a_max=1)
        return coord_features

    def hybrid_forward(self, F, x, coords):
        if isinstance(x, mx.nd.NDArray):
            self.xs = x.shape

        batch_size, rows, cols = self.xs[0], self.xs[2], self.xs[3]
        coord_features = self.get_coord_features(F, coords, rows, cols, batch_size,
                                                 **self._ctx_kwarg(x))
        return F.concat(coord_features, x, dim=1)


class ExtractQueryFeatures(gluon.HybridBlock):
    def __init__(self, extraction_method='ROIPooling', spatial_scale=1.0, eps=1e-4):
        super(ExtractQueryFeatures, self).__init__()
        self.cshape = None
        self.extraction_method = extraction_method
        self.spatial_scale = spatial_scale
        self.eps = eps

    def _ctx_kwarg(self, x):
        if isinstance(x, mx.nd.NDArray):
            return {"ctx": x.context}
        return {}

    def hybrid_forward(self, F, x, coords):
        if isinstance(coords, mx.nd.NDArray):
            self.cshape = coords.shape

        batch_size, num_points = self.cshape[0], self.cshape[1]
        ctx_kwarg = self._ctx_kwarg(coords)

        coords = F.reverse(F.reshape(coords, shape=(-1, 2)), axis=1)
        if self.extraction_method == 'ROIAlign':
            coords = coords - 0.5 / self.spatial_scale
            coords2 = coords
        else:
            coords2 = coords
        rois = F.concat(coords, coords2, dim=1)

        bi = F.arange(0, batch_size, **ctx_kwarg)
        bi = F.repeat(bi, num_points, axis=0)
        bi = F.reshape(bi, shape=(-1, 1))
        rois = F.concat(bi, rois)

        if self.extraction_method == 'ROIPooling':
            w = F.ROIPooling(x, rois,
                             pooled_size=(1, 1), spatial_scale=self.spatial_scale)
        elif self.extraction_method == 'ROIAlign':
            w = F.contrib.ROIAlign(x, rois,
                             pooled_size=(1, 1), spatial_scale=self.spatial_scale)
        else:
            assert False

        w = F.reshape(w, shape=(0, -1))

        return w


class AverageMaskedFeatures(gluon.HybridBlock):
    def __init__(self, spatial_scale=1.0):
        super(AverageMaskedFeatures, self).__init__()
        self.xshape = None
        self.mshape = None
        self.spatial_scale = spatial_scale

    def hybrid_forward(self, F, x, masks):
        if isinstance(masks, mx.nd.NDArray):
            self.xshape = x.shape
            self.mshape = masks.shape

        batch_size = self.xshape[0]
        num_masks = self.mshape[0]
        num_points = num_masks // batch_size

        masks = F.expand_dims(masks, axis=1)
        if self.spatial_scale < 1.0:
            masks = mx.nd.contrib.BilinearResize2D(masks,
                                                   height=self.xshape[2], width=self.xshape[3])

        xr = F.repeat(x, num_points, axis=0)
        xr = F.broadcast_mul(xr, masks)

        ws = F.sum(xr, axis=(0, 1), exclude=True)
        masks_sum = F.sum(masks, axis=(0, 1), exclude=True) + 1e-6
        w = F.broadcast_div(ws, masks_sum)

        return w
