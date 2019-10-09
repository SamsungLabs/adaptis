import torch
import torchvision
import torch.nn as nn

import adaptis.model.initializer as initializer


def select_activation_function(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return nn.ReLU
        elif activation.lower() == 'softplus':
            return nn.Softplus
        else:
            raise ValueError(f"Unknown activation type {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError(f"Unknown activation type {activation}")


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale

        super().__init__(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale,
            padding=1,
            groups=groups,
            bias=False)

        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class AdaIN(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super(AdaIN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm

        self.affine_scale = nn.Linear(in_channels, out_channels, bias=True)
        self.affine_bias = nn.Linear(in_channels, out_channels, bias=True)

        if norm:
            self.in_2d = nn.InstanceNorm2d(in_channels, affine=False, momentum=0.0, track_running_stats=False)

    def forward(self, x, w):
        y_scale = self.affine_scale(w)[:, :, None, None]
        # + 1 compensates zero initialization
        y_bias = 1 + self.affine_bias(w)[:, :, None, None]

        if self.norm:
            x = self.in_2d(x)

        x_scaled = (x * y_scale) + y_bias
        return x_scaled


class ExtractQueryFeatures(nn.Module):
    def __init__(self, extraction_method='ROIPooling', spatial_scale=1.0, eps=1e-4):
        super(ExtractQueryFeatures, self).__init__()
        assert extraction_method in ['ROIPooling', 'ROIAlign']
        if extraction_method == 'ROIPooling':
            self.shift_transform = lambda x: x
            self.extractor = torchvision.ops.RoIPool(output_size=(1, 1), spatial_scale=spatial_scale)
        elif extraction_method == 'ROIAlign':
            self.shift_transform = lambda x: x - 0.5 / self.spatial_scale
            self.extractor = torchvision.ops.RoIAlign(output_size=(1, 1), spatial_scale=spatial_scale,
                                                      sampling_ratio=-1)

        self.extraction_method = extraction_method
        self.spatial_scale = spatial_scale
        self.eps = eps

    def forward(self, features, points):
        batch_size, num_points = points.size()[:2]
        points = torch.flip(points.view(batch_size, num_points, 2), [2])
        # coords is batch_size * num_points x 4
        coords_from = self.shift_transform(points)
        coords_to = points
        coords = torch.cat((coords_from, coords_to), dim=2)

        # rois is batch_size*num_points x 5
        batch_index = torch.arange(0, batch_size).unsqueeze(1)
        batch_index = batch_index.repeat_interleave(num_points, dim=1).unsqueeze(2).float().to(points.device)
        rois = torch.cat((batch_index, coords), dim=2).view(batch_size * num_points, 5)

        w = self.extractor(features, rois)

        return w.view(batch_size * num_points, -1)


class AppendCoordFeatures(nn.Module):
    def __init__(self, norm_radius, append_dist=True, spatial_scale=1.0):
        super(AppendCoordFeatures, self).__init__()

        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.append_dist = append_dist

    def _get_coord_features(self, points, size):
        batch_size, _, h, w = size
        rows = torch.arange(0, h).view(1, 1, h, 1).repeat_interleave(w, dim=3)
        cols = torch.arange(0, w).view(1, 1, 1, w).repeat_interleave(h, dim=2)

        rows = rows.repeat_interleave(batch_size, dim=0)
        cols = cols.repeat_interleave(batch_size, dim=0)

        coords = torch.cat((rows, cols), dim=1).float().to(points.device)  # batch_size*num_points x 2 x h x w

        add_xy = (points * self.spatial_scale).view(batch_size, 2, 1)
        add_xy = add_xy.repeat_interleave(h * w, dim=2).view(batch_size, 2, h, w)

        relative_coords = (coords - add_xy) / (self.norm_radius * self.spatial_scale)

        if self.append_dist:
            dist = torch.sqrt((relative_coords ** 2).sum(1, keepdim=True))  # batch_size*num_points x 1 x h x w
            relative_coords = torch.cat((relative_coords, dist), dim=1)  # batch_size*num_points x 3 x h x w

        return relative_coords.clamp(-1, 1)

    def forward(self, x, points):
        size = x.size()
        coord_features = self._get_coord_features(points, size)

        return torch.cat((coord_features, x), dim=1)
