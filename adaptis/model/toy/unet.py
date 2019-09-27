import mxnet as mx
from mxnet import gluon


class UNet(gluon.nn.HybridBlock):
    def __init__(self, num_blocks=4, first_channels=64, max_width=512,
                 norm_layer=gluon.nn.BatchNorm, **kwargs):
        super(UNet, self).__init__(**kwargs)

        self.num_blocks = num_blocks
        with self.name_scope():
            self.image_bn = norm_layer(in_channels=3)

            prev_ch = 0
            for i in range(num_blocks + 1):
                if i == 0:
                    dblock = down_block(first_channels, norm_layer=norm_layer)
                else:
                    dblock = gluon.nn.HybridSequential()
                    dblock_ch = min(first_channels * (2 ** i), max_width)
                    dblock.add(
                        gluon.nn.MaxPool2D(2, 2, ceil_mode=True),
                        down_block(dblock_ch, norm_layer=norm_layer)
                    )
                    prev_ch = dblock_ch
                setattr(self, f'd{i}', dblock)

            for i in range(num_blocks):
                uindx = self.num_blocks - i - 1
                block_width = first_channels * (2 ** uindx)
                block_ch = min(block_width, max_width)
                block_shrink = uindx != 0 and block_width <= max_width
                ublock = up_block(block_ch, shrink=block_shrink,
                                  in_channels=prev_ch,
                                  norm_layer=norm_layer)
                prev_ch = block_ch
                if block_shrink:
                    prev_ch //= 2
                setattr(self, f'u{uindx}', ublock)

            self.fe = gluon.nn.HybridSequential()
            self.fe.add(
                gluon.nn.Conv2D(channels=first_channels, kernel_size=3, padding=1, activation='relu'),
                norm_layer(in_channels=first_channels),
                gluon.nn.Conv2D(channels=first_channels, kernel_size=3, padding=1, activation='relu'),
                norm_layer(in_channels=first_channels)
            )

    def hybrid_forward(self, F, x):
        x = self.image_bn(x)

        d_outs = []
        for i in range(self.num_blocks + 1):
            x = getattr(self, f'd{i}')(x)
            d_outs.append(x)

        for i in range(self.num_blocks):
            u_indx = self.num_blocks - i - 1
            x = getattr(self, f'u{u_indx}')(x, d_outs[u_indx])

        return self.fe(x),


def down_block(channels, norm_layer=gluon.nn.BatchNorm):
    out = gluon.nn.HybridSequential()

    out.add(
        ConvBlock(channels, 3, norm_layer=norm_layer),
        ConvBlock(channels, 3, norm_layer=norm_layer)
    )
    return out


class up_block(gluon.nn.HybridBlock):
    def __init__(self, channels, shrink=True, norm_layer=gluon.nn.BatchNorm, in_channels=0, **kwargs):
        super(up_block, self).__init__(**kwargs)

        self.upsampler = gluon.nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2,
                                                  in_channels=in_channels,
                                                  padding=1, use_bias=False, groups=channels,
                                                  weight_initializer=mx.init.Bilinear())
        self.upsampler.collect_params().setattr('grad_req', 'null')

        self.conv1 = ConvBlock(channels, 1, norm_layer=norm_layer)
        self.conv3_0 = ConvBlock(channels, 3, norm_layer=norm_layer)
        if shrink:
            self.conv3_1 = ConvBlock(channels // 2, 3, norm_layer=norm_layer)
        else:
            self.conv3_1 = ConvBlock(channels, 3, norm_layer=norm_layer)

    def hybrid_forward(self, F, x, s):
        x = self.upsampler(x)
        x = self.conv1(x)
        x = F.relu(x)

        x = F.Crop(*[x, s], center_crop=True)
        x = F.concat(s, x, dim=1)

        x = self.conv3_0(x)
        x = self.conv3_1(x)

        return x


def ConvBlock(channels, kernel_size, norm_layer=gluon.nn.BatchNorm):
    out = gluon.nn.HybridSequential()

    out.add(
        gluon.nn.Conv2D(channels, kernel_size, padding=kernel_size // 2, use_bias=False),
        gluon.nn.Activation('relu'),
        norm_layer()
    )
    return out
