import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from .resnet import ResNetBackbone


class ResNetFPN(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet50', backbone_lr_mult=0.1, **kwargs):
        super(ResNetFPN, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = _FPNHead(output_channels=256, **kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        p1, p2, p3, p4 = self.head(c1, c2, c3, c4)

        return p1, p2, p3, p4


class _FPNHead(HybridBlock):
    def __init__(self, output_channels=256, norm_layer=nn.BatchNorm):
        super(_FPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block3 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block2 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        p4 = self.block4(c4)
        p3 = self._resize_as(F, 'id_1', p4, c3) + self.block3(c3)
        p2 = self._resize_as(F, 'id_2', p3, c2) + self.block2(c2)
        p1 = self._resize_as(F, 'id_3', p2, c1) + self.block1(c1)

        return p1, p2, p3, p4

    def _resize_as(self, F, name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key] = h
            self._hdsize[w_key] = w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x, h, w)


class SemanticFPNHead(HybridBlock):
    def __init__(self, num_classes, output_channels=128, norm_layer=nn.BatchNorm):
        super(SemanticFPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4_1 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)
            self.block4_2 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)
            self.block4_3 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)

            self.block3_1 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)
            self.block3_2 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)

            self.block2 = ConvBlock(output_channels, kernel_size=3, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

            self.postprocess_block = nn.Conv2D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        out4 = self._resize_as(F, 'id_1', self.block4_1(c4), c3)
        out4 = self._resize_as(F, 'id_2', self.block4_2(out4), c2)
        out4 = self._resize_as(F, 'id_3', self.block4_3(out4), c1)

        out3 = self._resize_as(F, 'id_4', self.block3_1(c3), c2)
        out3 = self._resize_as(F, 'id_5', self.block3_2(out3), c1)

        out2 = self._resize_as(F, 'id_6', self.block2(c2), c1)

        out1 = self.block1(c1)

        out = out1 + out2 + out3 + out4

        out = self.postprocess_block(out)

        return out

    def _resize_as(self, F,name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key]=h
            self._hdsize[w_key]=w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x,h,w)


class ConvBlock(HybridBlock):
    def __init__(self, output_channels, kernel_size, activation='relu', norm_layer=nn.BatchNorm):
        super().__init__()
        self.body = nn.HybridSequential()
        self.body.add(
            nn.Conv2D(output_channels, kernel_size=kernel_size, activation=activation),
            norm_layer(in_channels=output_channels)
        )

    def hybrid_forward(self, F, x):
        return self.body(x)
