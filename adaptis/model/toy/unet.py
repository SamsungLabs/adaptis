import torch
import torch.nn as nn

import adaptis.model.ops as ops


class UNet(nn.Module):
    def __init__(self, num_blocks, first_channels=64, max_width=512, norm_layer=nn.BatchNorm2d, train_upsampling=True):
        super(UNet, self).__init__()

        self.num_blocks = num_blocks
        self.image_bn = norm_layer(3)
        prev_ch = 3

        # encoder
        self.encoder = nn.ModuleList()
        for idx in range(num_blocks + 1):
            out_ch = min(first_channels * (2 ** idx), max_width)
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2, 2, ceil_mode=True) if idx > 0 else nn.Identity(),
                DownBlock(prev_ch, out_ch, norm_layer=norm_layer)
            ))
            prev_ch = out_ch

        # decoder
        self.decoder = nn.ModuleList()
        for idx in reversed(range(num_blocks)):
            block_width = first_channels * (2 ** idx)
            block_ch = min(block_width, max_width)
            block_shrink = (idx > 0) and (block_width <= max_width)

            self.decoder.append(
                UpBlock(prev_ch, block_ch, shrink=block_shrink, norm_layer=norm_layer, train_upsampling=train_upsampling)
            )

            prev_ch = block_ch // 2 if block_shrink else block_ch

        self.final_block = DownBlock(prev_ch, first_channels, kernel_size=3, norm_layer=norm_layer)
        self.feature_channels = first_channels

    def get_feature_channels(self):
        return self.feature_channels

    def forward(self, x):
        x = self.image_bn(x)

        encoder_output = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_output.append(x)

        for idx, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, encoder_output[-idx - 2])

        x = self.final_block(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shrink=True, norm_layer=nn.BatchNorm2d, train_upsampling=False):
        super(UpBlock, self).__init__()
        if train_upsampling:
            self.upsampler = nn.Sequential(
                ops.BilinearConvTranspose2d(in_channels, out_channels,  scale=2, groups=out_channels),
                nn.ReLU()
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.ReLU()
            )

        self.conv3_0 = ConvBlock(2 * out_channels, out_channels, 3, norm_layer=norm_layer)
        if shrink:
            self.conv3_1 = ConvBlock(out_channels, out_channels // 2, 3, norm_layer=norm_layer)
        else:
            self.conv3_1 = ConvBlock(out_channels, out_channels, 3, norm_layer=norm_layer)

    def forward(self, x, skip):
        x = self.upsampler(x)

        if skip.size(2) > x.size(2):
            skip = _center_crop(skip, x.size()[2:])
        elif skip.size(2) < x.size(2):
            x = _center_crop(x, skip.size()[2:])

        x = torch.cat((x, skip), dim=1)

        x = self.conv3_0(x)
        x = self.conv3_1(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=nn.BatchNorm2d):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, norm_layer),
            ConvBlock(out_channels, out_channels, kernel_size, norm_layer)
        )

    def forward(self, x):
        return self.down_block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.ReLU(),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity()
        )

    def forward(self, x):
        return self.conv_block(x)


def _center_crop(tensor, target_size):
    _, _, tensor_h, tensor_w = tensor.size()
    diff_h = (tensor_h - target_size[0])
    diff_w = (tensor_w - target_size[1])

    from_h, from_w = diff_h // 2, diff_w // 2
    to_h = target_size[0] + from_h
    to_w = target_size[1] + from_w
    return tensor[:, :, from_h: to_h, from_w: to_w]
