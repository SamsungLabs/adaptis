import mxnet as mx
from mxnet import gluon

from adaptis.model.cityscapes.deeplab_v3 import DeepLabV3Plus
from adaptis.model.adaptis import AdaptIS
from adaptis.model.ops import AdaIN, ExtractQueryFeatures, AppendCoordFeatures
from adaptis.model.basic_blocks import SepConvHead, FCController, SeparableConv2D
from .resnet_fpn import SemanticFPNHead, ResNetFPN


def get_cityscapes_model(num_classes, norm_layer, backbone='resnet50',
                         with_proposals=False):
    model = AdaptIS(
        feature_extractor=DeepLabV3Plus(backbone=backbone, norm_layer=norm_layer),
        adaptis_head=CityscapesAdaptISHead(
            FCController(3 * [128], norm_layer=norm_layer),
            ch=128, norm_radius=280,
            spatial_scale=1.0/4.0,
            norm_layer=norm_layer
        ),
        segmentation_head=SepConvHead(num_classes, channels=192, in_channels=256, num_layers=2, norm_layer=norm_layer),
        proposal_head=SepConvHead(1, channels=128, in_channels=256, num_layers=2,
                                  dropout_ratio=0.5, dropout_indx=0, norm_layer=norm_layer),
        with_proposals=with_proposals,
        spatial_scale=1.0/4.0
    )

    return model


def get_fpn_model(num_classes, norm_layer, backbone='resnet50',
                         with_proposals=False):
    model = AdaptIS(
        feature_extractor=ResNetFPN(backbone=backbone, norm_layer=norm_layer),
        adaptis_head=CityscapesAdaptISHead(
            FCController(3 * [128], norm_layer=norm_layer),
            ch=128, norm_radius=280,
            spatial_scale=1.0/4.0,
            norm_layer=norm_layer
        ),
        segmentation_head=SemanticFPNHead(num_classes, output_channels=256, norm_layer=norm_layer),
        proposal_head=SepConvHead(1, channels=128, in_channels=256, num_layers=2,
                                  dropout_ratio=0.5, dropout_indx=0, norm_layer=norm_layer),
        with_proposals=with_proposals,
        spatial_scale=1.0/4.0
    )
    return model


class CityscapesAdaptISHead(gluon.HybridBlock):
    def __init__(self, controller_net, ch=128, norm_radius=190, spatial_scale=0.25,
                 norm_layer=gluon.nn.BatchNorm):
        super(CityscapesAdaptISHead, self).__init__()

        self.num_points = None
        with self.name_scope():
            self.eqf = ExtractQueryFeatures(extraction_method='ROIAlign', spatial_scale=spatial_scale)
            self.controller_net = controller_net

            self.add_coord_features = AppendCoordFeatures(norm_radius=norm_radius, spatial_scale=spatial_scale)

            self.block0 = gluon.nn.HybridSequential()
            self.block0.add(
                gluon.nn.Conv2D(channels=ch, kernel_size=3, padding=1, activation='relu'),
                norm_layer(in_channels=ch),
                SeparableConv2D(ch, in_channels=ch, dw_kernel=3, dw_padding=1,
                                norm_layer=norm_layer, activation='relu'),
                gluon.nn.Conv2D(channels=ch, kernel_size=1, padding=0),
                gluon.nn.LeakyReLU(0.2)
            )

            self.adain = AdaIN(ch)

            self.block1 = gluon.nn.HybridSequential()
            for i in range(3):
                self.block1.add(
                    SeparableConv2D(ch // (2 ** i), in_channels=min(ch, 2 * ch // (2 ** i)),
                                    dw_kernel=3, dw_padding=1,
                                    norm_layer=norm_layer, activation='relu'),
                )
            self.block1.add(gluon.nn.Conv2D(channels=1, kernel_size=1))

    def hybrid_forward(self, F, p1_features, points):
        adaptive_input, controller_input = self.get_point_invariant_states(F, p1_features)
        return self.get_instances_maps(F, points, adaptive_input, controller_input)

    def get_point_invariant_states(self, F, backbone_features):
        adaptive_input = backbone_features

        if getattr(self.controller_net, 'return_map', False):
            controller_input = self.controller_net(backbone_features)
        else:
            controller_input = backbone_features

        return adaptive_input, controller_input

    def get_instances_maps(self, F, points, adaptive_input, controller_input):
        if isinstance(points, mx.nd.NDArray):
            self.num_points = points.shape[1]

        if getattr(self.controller_net, 'return_map', False):
            w = self.eqf(controller_input, points)
        else:
            w = self.eqf(controller_input, points)
            w = self.controller_net(w)

        points = F.reshape(points, shape=(-1, 2))
        x = F.repeat(adaptive_input, self.num_points, axis=0)
        x = self.add_coord_features(x, points)

        x = self.block0(x)
        x = self.adain(x, w)
        x = self.block1(x)

        return x
