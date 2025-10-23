import torch
import torch.nn as nn
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class RKAN_DenseNet(nn.Module):
    def __init__(self, num_classes = 1000, version = "densenet121", kan_type = "chebyshev", pretrained = False, reduce_factor = [2, 2, 2, 2],
                 n_convs = 1, mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4):
        super(RKAN_DenseNet, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor
        
        if pretrained:
            self.densenet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.densenet = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")
        
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        layer_config = {
            "densenet121": [256, 512, 1024, 1024],
            "densenet169": [256, 512, 1280, 1664],
            "densenet201": [256, 512, 1792, 1920],
            "densenet161": [384, 768, 2112, 2208]
        }
        channels = layer_config[version]

        # KAN convolutions for each stage
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 3 else (2, 2), padding = (1, 1),
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
                nn.Conv2d(channels[0] // 2, channels[0] // 2 * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1] // 2, channels[1] // 2 * self.inv_factor, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2] // 2, channels[2] // 2 * self.inv_factor, kernel_size = 1, stride = 1, bias = False)
            ])
            self.conv_expand = nn.ModuleList([
                nn.Conv2d(64 * self.inv_factor, channels[0] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0] // 2 * self.inv_factor, channels[1] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1] // 2 * self.inv_factor, channels[2] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2] // 2 * self.inv_factor, channels[3], kernel_size = 1, stride = 1, bias = False)
            ])
        else:
            self.conv_reduce = nn.ModuleList([
                nn.Conv2d(64, 64 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0] // 2, channels[0] // 2 // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1] // 2, channels[1] // 2 // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2] // 2, channels[2] // 2 // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
            ])

            self.conv_expand = nn.ModuleList([
                nn.Conv2d(64 // reduce_factor[0], channels[0] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[0] // 2 // reduce_factor[1], channels[1] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[1] // 2 // reduce_factor[2], channels[2] // 2, kernel_size = 1, stride = 1, bias = False),
                nn.Conv2d(channels[2] // 2 // reduce_factor[3], channels[3], kernel_size = 1, stride = 1, bias = False)
            ])

        # KAN normalization
        if self.inv_bottleneck:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d((ch // 2) * self.inv_factor) if i > 0 else nn.BatchNorm2d(ch * self.inv_factor) for i, ch in enumerate([64] + channels[:-1])])
        else:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d((ch // 2) // reduce_factor[i]) if i > 0 else nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch // 2) if i < len(channels) - 1 else nn.BatchNorm2d(ch) for i, ch in enumerate(channels)])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        self.se_blocks = nn.ModuleList([self._make_se_block(ch // 2) if i < len(channels) - 1 else self._make_se_block(ch) for i, ch in enumerate(channels)])

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
        out = self.densenet.features.conv0(x)
        out = self.densenet.features.norm0(out)
        out = self.densenet.features.relu0(out)
        out = self.densenet.features.pool0(out)
        
        dense_blocks = [self.densenet.features.denseblock1, self.densenet.features.denseblock2, self.densenet.features.denseblock3, self.densenet.features.denseblock4]
        transition_layers = [self.densenet.features.transition1, self.densenet.features.transition2, self.densenet.features.transition3]
        
        for i, (block, mechanism) in enumerate(zip(dense_blocks, self.mechanisms)):
            identity = out
            out = block(out)
            
            # Apply norm5 and ReLU before adding residual for the last denseblock if post-transition
            if i == len(dense_blocks) - 1:
                out = self.densenet.features.norm5(out)

            if i < len(transition_layers):
                out = transition_layers[i](out)

            # Post-transition mechanism
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

        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out