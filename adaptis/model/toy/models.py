import mxnet as mx
from mxnet import gluon

from adaptis.model.adaptis import AdaptIS
from .unet import UNet
from adaptis.model.basic_blocks import ConvHead, SimpleConvController
from adaptis.model.ops import AdaIN, ExtractQueryFeatures, AppendCoordFeatures


def get_unet_model(norm_layer, channel_width=32, max_width=512, with_proposals=False, rescale_output=(0.2, -1.7)):
    return AdaptIS(
        feature_extractor=UNet(num_blocks=4, first_channels=channel_width, max_width=max_width,
                               norm_layer=norm_layer),
        adaptis_head=ToyAdaptISHead(
            SimpleConvController(3, channel_width, norm_layer=norm_layer),
            ch=channel_width, norm_radius=42, with_coord_features=True,
            norm_layer=norm_layer,
            rescale_output=rescale_output
        ),
        segmentation_head=ConvHead(2, channels=32, num_layers=3, norm_layer=norm_layer),
        proposal_head=ConvHead(1, channels=32, num_layers=2, norm_layer=norm_layer),
        with_proposals=with_proposals
    )


class ToyAdaptISHead(gluon.HybridBlock):
    def __init__(self, controller_net, ch=32, norm_radius=42,
                 with_coord_features=True, norm_layer=gluon.nn.BatchNorm,
                 rescale_output=None):
        super(ToyAdaptISHead, self).__init__()

        self.num_points = None
        self.with_coord_features = with_coord_features
        self.rescale_output = rescale_output

        with self.name_scope():
            self.eqf = ExtractQueryFeatures(extraction_method='ROIAlign', spatial_scale=1.0)
            self.controller_net = controller_net

            if self.with_coord_features:
                self.add_coord_features = AppendCoordFeatures(norm_radius=norm_radius, spatial_scale=1.0)

            self.block0 = gluon.nn.HybridSequential()
            for i in range(3):
                self.block0.add(
                    gluon.nn.Conv2D(channels=ch, kernel_size=3, padding=1, activation='relu'),
                    norm_layer(in_channels=ch)
                )

            self.adain = AdaIN(ch)

            self.block1 = gluon.nn.HybridSequential()
            for i in range(1):
                self.block1.add(
                    gluon.nn.Conv2D(channels=ch // 2, kernel_size=3, padding=1, activation='relu'),
                    norm_layer(in_channels=ch // 2)
                )
            self.block1.add(gluon.nn.Conv2D(channels=1, kernel_size=1))

    def hybrid_forward(self, F, backbone_features, points):
        adaptive_input, controller_input = self.get_point_invariant_states(F, backbone_features)
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
        if self.with_coord_features:
            x = self.add_coord_features(x, points)

        x = self.block0(x)
        x = self.adain(x, w)
        x = self.block1(x)

        if self.rescale_output:
            scale, bias = self.rescale_output
            x = scale * x + bias

        return x
