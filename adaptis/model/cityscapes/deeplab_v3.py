from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from adaptis.model.basic_blocks import SeparableConv2D
from .resnet import ResNetBackbone


class DeepLabV3Plus(gluon.HybridBlock):
    def __init__(self, backbone='resnet50', backbone_lr_mult=0.1, **kwargs):
        super(DeepLabV3Plus, self).__init__()

        self._c1_shape = None
        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, **kwargs)

            self.head = _DeepLabHead(256, in_filters=256 + 32, **kwargs)
            self.skip_project = _SkipProject(32, **kwargs)
            self.aspp = _ASPP(2048, [12, 24, 36], **kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self, F, x):
        c1, _, c3, c4 = self.backbone(x)
        c1 = self.skip_project(c1)

        if hasattr(c1, 'shape'):
            self._c1_shape = c1.shape

        x = self.aspp(c4)
        x = F.contrib.BilinearResize2D(x, height=self._c1_shape[2], width=self._c1_shape[3])
        x = F.concat(x, c1, dim=1)
        x = self.head(x)

        return x,


class _SkipProject(HybridBlock):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm):
        super(_SkipProject, self).__init__()

        with self.name_scope():
            self.skip_project = nn.HybridSequential()
            self.skip_project.add(nn.Conv2D(out_channels, kernel_size=1, use_bias=False))
            self.skip_project.add(norm_layer(in_channels=out_channels))
            self.skip_project.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        return self.skip_project(x)


class _DeepLabHead(HybridBlock):
    def __init__(self, output_channels, in_filters, norm_layer=nn.BatchNorm):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()

            self.block.add(SeparableConv2D(256, in_channels=in_filters, dw_kernel=3, dw_padding=1,
                                           activation='relu', norm_layer=norm_layer))
            self.block.add(SeparableConv2D(256, in_channels=256, dw_kernel=3, dw_padding=1,
                                           activation='relu', norm_layer=norm_layer))

            self.block.add(nn.Conv2D(channels=output_channels,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        return self.block(x)


class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, out_channels=256,
                 project_dropout=0.5,
                 norm_layer=nn.BatchNorm):
        super(_ASPP, self).__init__()

        b0 = nn.HybridSequential()
        with b0.name_scope():
            b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                             kernel_size=1, use_bias=False))
            b0.add(norm_layer(in_channels=out_channels))
            b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        with self.concurent.name_scope():
            self.concurent.add(b0)
            self.concurent.add(b1)
            self.concurent.add(b2)
            self.concurent.add(b3)
            self.concurent.add(b4)

        self.project = nn.HybridSequential()
        with self.project.name_scope():
            self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                       kernel_size=1, use_bias=False))
            self.project.add(norm_layer(in_channels=out_channels))
            self.project.add(nn.Activation("relu"))
            if project_dropout > 0:
                self.project.add(nn.Dropout(project_dropout))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))


class _AsppPooling(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self._out_h = None
        self._out_w = None
        self.gap = nn.HybridSequential()
        with self.gap.name_scope():
            self.gap.add(nn.GlobalAvgPool2D())
            self.gap.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
            self.gap.add(norm_layer(in_channels=out_channels))
            self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        if hasattr(x, 'shape'):
            _, _, h, w = x.shape
            self._out_h = h
            self._out_w = w
        else:
            h, w = self._out_h, self._out_w
            assert h is not None

        pool = self.gap(x)
        return F.contrib.BilinearResize2D(pool, height=h, width=w)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels))
        block.add(nn.Activation('relu'))
    return block
