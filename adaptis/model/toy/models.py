import torch.nn as nn

from adaptis.model import ops
from adaptis.model.toy.unet import UNet
from adaptis.model.adaptis import AdaptIS
from adaptis.model import basic_blocks


def get_unet_model(channel_width=32, max_width=512, with_proposals=False, rescale_output=(0.2, -1.7), norm_layer=nn.BatchNorm2d):
    unet = UNet(
        num_blocks=4,
        first_channels=channel_width,
        max_width=max_width,
        norm_layer=norm_layer
    )
    in_channels = unet.get_feature_channels()

    return AdaptIS(
        backbone=unet,
        adaptis_head=ToyAdaptISHead(
            basic_blocks.SimpleConvController(3, in_channels, channel_width, norm_layer=norm_layer),
            in_channels,
            channels=channel_width,
            norm_radius=42,
            with_coord_features=True,
            rescale_output=rescale_output,
            norm_layer=norm_layer
        ),
        segmentation_head=basic_blocks.ConvHead(2, in_channels=in_channels, num_layers=3, norm_layer=norm_layer),
        proposal_head=basic_blocks.ConvHead(1, in_channels=in_channels, num_layers=2, norm_layer=norm_layer),
        with_proposals=with_proposals
    )


class ToyAdaptISHead(nn.Module):
    def __init__(self, controller_network, in_channels, channels=32, norm_radius=42, with_coord_features=True,
                 norm_layer=nn.BatchNorm2d, rescale_output=None):
        super(ToyAdaptISHead, self).__init__()

        self.num_points = None
        self.with_coord_features = with_coord_features
        self.rescale_output = rescale_output

        self.EQF = ops.ExtractQueryFeatures(extraction_method='ROIAlign', spatial_scale=1.0)
        self.controller = controller_network
        if with_coord_features:
            self.add_coord_features = ops.AppendCoordFeatures(norm_radius, spatial_scale=1.0)
            in_channels += 2
            if self.add_coord_features.append_dist:
                in_channels += 1

        block0 = []
        for _ in range(3):
            block0.extend([
                nn.Conv2d(in_channels, channels, 3, padding=1),
                nn.ReLU(),
                norm_layer(channels)
            ])
            in_channels = channels
        self.block0 = nn.Sequential(*block0)

        self.adain = ops.AdaIN(channels, channels)

        block1 = []
        for _ in range(1):
            block1.extend([
                nn.Conv2d(in_channels, channels // 2, 3, padding=1),
                nn.ReLU(),
                norm_layer(channels // 2)
            ])
            in_channels = channels // 2
        block1.append(nn.Conv2d(channels // 2, 1, 1))
        self.block1 = nn.Sequential(*block1)

    def _get_point_invariant_features(self, backbone_features):
        adaptive_input = backbone_features

        if getattr(self.controller, 'return_map', False):
            controller_input = self.controller(backbone_features)
        else:
            controller_input = backbone_features

        return adaptive_input, controller_input

    def _get_instance_maps(self, points, adaptive_input, controller_input):
        self.num_points = points.size(1)
        if getattr(self.controller, 'return_map', False):
            w = self.EQF(controller_input, points)
        else:
            w = self.EQF(controller_input, points)
            w = self.controller(w)

        points = points.view(-1, 2)
        x = adaptive_input.repeat_interleave(self.num_points, dim=0)
        if self.with_coord_features:
            x = self.add_coord_features(x, points)
        x = self.block0(x)
        x = self.adain(x, w)
        x = self.block1(x)

        if self.rescale_output:
            scale, bias = self.rescale_output
            x = scale * x + bias

        return x

    def forward(self, backbone_features, points):
        adaptive_input, controller_input = self._get_point_invariant_features(backbone_features)
        return self._get_instance_maps(points, adaptive_input, controller_input)
