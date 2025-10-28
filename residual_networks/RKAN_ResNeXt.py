import torch
import torch.nn as nn
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class RKANeXt(nn.Module):
    def __init__(self, num_classes = 1000, version = "resnext50_32x4d", kan_type = "chebyshev", pretrained = False, reduce_factor = [2, 2, 2, 2],
                 n_convs = 1, mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4):
        super(RKANeXt, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor
        
        if pretrained:
            self.resnext = getattr(models, version)(weights = "DEFAULT")
        else:
            self.resnext = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)
        layer_config = {
            "resnext50_32x4d": [256, 512, 1024, 2048],
            "resnext101_32x8d": [256, 512, 1024, 2048],
            "resnext101_64x4d": [256, 512, 1024, 2048]
        }
        channels = layer_config[version]

        # KAN convolutions for each stage
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 0 else (2, 2), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[0], grid_size = grid_size[0])
            for i in range(len(channels))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[1], grid_size = grid_size[1])
            for i in range(len(channels))
        ])

        # Bottleneck for KAN
        if self.inv_bottleneck:
            self.conv_reduce = nn.ModuleList([
                nn.Conv2d(64, 64 * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[0] * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1], channels[1] * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2], channels[2] * self.inv_factor, kernel_size = 1, stride = 1, bias = False)
            ])

            self.conv_expand = nn.ModuleList([
                nn.Conv2d((64 * self.inv_factor) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[0] * self.inv_factor) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[1] * self.inv_factor) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[2] * self.inv_factor) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
            ])
        else:
            self.conv_reduce = nn.ModuleList([
                nn.Conv2d(64, 64 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[0] // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1], channels[1] // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2], channels[2] // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
            ])

            self.conv_expand = nn.ModuleList([
                nn.Conv2d((64 // reduce_factor[0]) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[0] // reduce_factor[1]) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[1] // reduce_factor[2]) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d((channels[2] // reduce_factor[3]) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
            ])

        # KAN normalization
        bn_params = {"momentum": 0.1, "eps": 1e-5, "affine": True}
        if self.inv_bottleneck:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch * self.inv_factor, **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        else:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i], **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])
        
    def _make_se_block(self, channels, reduction = 16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
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
        out = self.resnext.conv1(x)
        out = self.resnext.bn1(out)
        out = self.resnext.relu(out)
        out = self.resnext.maxpool(out)
        
        layers = [self.resnext.layer1, self.resnext.layer2, self.resnext.layer3, self.resnext.layer4]
        for i, (layer, mechanism) in enumerate(zip(layers, self.mechanisms)):
            identity = out
            out = layer(out)

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

        out = self.resnext.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.resnext.fc(out)
        return out