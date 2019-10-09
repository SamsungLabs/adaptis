import torch.nn as nn

import adaptis.model.ops as ops


class ConvHead(nn.Module):
    def __init__(self, out_channels, in_channels=32, num_layers=1, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []

        for i in range(num_layers):
            convhead.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
                nn.ReLU(),
                norm_layer(in_channels) if norm_layer is not None else nn.Identity()
            ])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))

        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class FCController(nn.Module):
    def __init__(self, input_size, layers_sizes,  activation='relu', norm_layer=nn.BatchNorm2d):
        super(FCController, self).__init__()

        # flag that indicates whether we use fully convolutional controller or not
        self.return_map = False

        # select activation function
        _activation = ops.select_activation_function(activation)

        controller = []
        for hidden_size in layers_sizes:
            controller.extend([
                nn.Linear(input_size, hidden_size),
                _activation(),
                norm_layer(hidden_size) if norm_layer is not None else nn.Identity()
            ])
            input_size = hidden_size
        self.controller = nn.Sequential(*controller)

    def forward(self, x):
        return self.controller(x)


class SimpleConvController(nn.Module):
    def __init__(self, num_layers, in_channels, latent_channels,
                 kernel_size=1, activation='relu', norm_layer=nn.BatchNorm2d):
        super(SimpleConvController, self).__init__()

        # flag that indicates whether we use fully convolutional controller or not
        self.return_map = True

        # select activation function
        _activation = ops.select_activation_function(activation)

        controller = []
        for i in range(num_layers):
            controller.extend([
                nn.Conv2d(in_channels, latent_channels, kernel_size),
                _activation(),
                norm_layer(latent_channels) if norm_layer is not None else nn.Identity()
            ])
        self.controller = nn.Sequential(*controller)

    def forward(self, x):
        x = self.controller(x)
        return x
