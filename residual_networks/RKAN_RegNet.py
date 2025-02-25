import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class RKAN_RegNet(nn.Module):
    def __init__(self, num_classes = 1000, version = "regnet_y_3_2gf", kan_type = "chebyshev", pretrained = False, reduce_factor = [2, 2, 2, 2], n_convs = 1,
                 mechanisms = [None, None, None, "addition"]):
        super(RKAN_RegNet, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor

        if pretrained:
            self.regnet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.regnet = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        self.regnet.fc = nn.Linear(self.regnet.fc.in_features, num_classes)
        layer_config = {
            "regnet_y_400mf": [48, 104, 208, 440],
            "regnet_y_800mf": [64, 144, 320, 784],
            "regnet_y_1_6gf": [48, 120, 336, 888],
            "regnet_x_3_2gf": [96, 192, 432, 1008],
            "regnet_y_3_2gf": [72, 216, 576, 1512],
            "regnet_x_8gf": [80, 240, 720, 1920],
            "regnet_y_8gf": [224, 448, 896, 2016],
            "regnet_y_16gf": [224, 448, 1232, 3024],
            "regnet_y_32gf": [232, 696, 1392, 3712]
        }
        channels = layer_config[version]

        # KAN convolutions for each layer
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), kan_type = kan_type, spline_order = 3)
            for i in range(len(channels))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), kan_type = kan_type, spline_order = 2)
            for i in range(len(channels))
        ])

        # Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(32, 32 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0], channels[0] // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1], channels[1] // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2], channels[2] // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand = nn.ModuleList([
            nn.Conv2d(32 // reduce_factor[0], channels[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0] // reduce_factor[1], channels[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1] // reduce_factor[2], channels[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2] // reduce_factor[3], channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        # KAN normalization
        self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([32] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()

        # Residual mechanisms
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])

    def _make_se_block(self, channels, reduction = 16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias = False),
            nn.Sigmoid()
    )

    def apply_mechanism(self, out, residual, layer_index, mechanism):
        if mechanism == "addition":
            return out + residual
        
        elif mechanism == "se":
            se_weight = self.se_blocks[layer_index](residual)
            return out + residual * se_weight
        
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}.")
    
    def forward(self, x):
        out = self.regnet.stem(x)
        for i, (block, mechanism) in enumerate(zip(self.regnet.trunk_output, self.mechanisms)):
            identity = out
            out = block(out)

            if mechanism is not None:
                residual = self.conv_reduce[i](identity)
                residual = self.silu(residual)
                
                residual = self.kan_conv1[i](residual)
                residual = self.kan_bn[i](residual)
                    
                residual = self.conv_expand[i](residual)
                residual = self.silu(residual)
                
                if i == len(self.mechanisms) - 1:
                    residual = self.kan_conv2[i](residual)
                residual = self.kan_expand_bn[i](residual)
                out = self.apply_mechanism(out, residual, i, mechanism)

        out = self.regnet.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.regnet.fc(out)
        return out