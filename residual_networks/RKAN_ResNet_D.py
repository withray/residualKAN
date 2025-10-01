import torch
import torch.nn as nn
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class RKANet_D(nn.Module):
    def __init__(self, num_classes = 1000, version = "resnetd50", kan_type = "chebyshev", pretrained = False, n_convs = 1, reduce_factor = [2, 2, 2, 2],
                 mechanisms = [None, None, None, "addition"], shortcut = False):
        super(RKANet_D, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.shortcut = shortcut

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")
        
        version_mapping = {f"resnetd{i}": f"resnet{i}" for i in [50, 101, 152]}
        backbone_version = version_mapping.get(version, version)

        if pretrained:
            base_resnet = getattr(models, backbone_version)(weights = "DEFAULT")
        else:
            base_resnet = getattr(models, backbone_version)(weights = None)

        # Three 3x3 convs instead of one 7x7
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )
        
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = self._make_resnetd_layer(base_resnet.layer1)
        self.layer2 = self._make_resnetd_layer(base_resnet.layer2)
        self.layer3 = self._make_resnetd_layer(base_resnet.layer3)
        self.layer4 = self._make_resnetd_layer(base_resnet.layer4)

        self.avgpool = base_resnet.avgpool
        self.fc = nn.Linear(base_resnet.fc.in_features, num_classes)
        layer_config = {
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048]
        }
        channels = layer_config[backbone_version]

        # KAN convolutions for each stage
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 0 else (2, 2), padding = (1, 1), kan_type = kan_type, spline_order = 3)
            for i in range(len(channels))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), kan_type = kan_type, spline_order = 2)
            for i in range(len(channels))
        ])

        # Bottleneck for KAN
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

        if self.shortcut:     
            self.shortcut_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])
            self.conv_shortcut = nn.ModuleList([
                nn.Conv2d(64, channels[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0], channels[1], kernel_size = 1, stride = 2, bias = False),
                nn.Conv2d(channels[1], channels[2], kernel_size = 1, stride = 2, bias = False),
                nn.Conv2d(channels[2], channels[3], kernel_size = 1, stride = 2, bias = False)
            ])

        # KAN normalization
        bn_params = {"momentum": 0.1, "eps": 1e-5, "affine": True}
        self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i], **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch, **bn_params) for ch in channels])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        self.gate_convs = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size = 1) for ch in channels])
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])
    
    def _make_resnetd_layer(self, layer):
        for block in layer:
            if hasattr(block, "downsample") and block.downsample is not None:
                original_downsample = block.downsample
                if len(original_downsample) >= 2:
                    conv = original_downsample[0]
                    bn = original_downsample[1]
                    if conv.stride[0] == 2:
                        block.downsample = nn.Sequential(
                            nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0),
                            nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size = 1, stride = 1, bias = False),
                            bn
                        )

            if hasattr(block, "conv1") and hasattr(block, "conv2"):
                if block.conv1.stride == (2, 2):
                    block.conv1.stride = (1, 1)
                    block.conv2.stride = (2, 2)
        return layer

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
        
        elif mechanism == "gating":
            gate = torch.sigmoid(self.gate_convs[layer_index](residual))
            return out * gate + residual * (1 - gate)
        
        elif mechanism == "se":
            se_weight = self.se_blocks[layer_index](residual)
            return out + residual * se_weight
        
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}.")

    def forward(self, x):
        out = self.stem(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
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

                if self.shortcut:
                    shortcut = self.conv_shortcut[i](identity)
                    shortcut = self.shortcut_bn[i](shortcut)
                    residual = residual + shortcut
                out = self.apply_mechanism(out, residual, i, mechanism)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out