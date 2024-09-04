import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class ConvKNeXt(nn.Module):
    def __init__(self, num_classes = 100, version = "convnext_tiny", kan_type = "chebyshev", pretrained = False, fcl = "convnext",
                 reduce_factor = [4, 4, 4, 4], grid_size = 5, n_convs = 1, single_conv = True, mechanisms = [None, None, None, None]):
        super(ConvKNeXt, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.single_conv = single_conv
        self.mechanisms = mechanisms
        self.fcl = fcl
        
        if pretrained:
            self.convnext = getattr(models, version)(weights = "DEFAULT")
        else:
            self.convnext = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        self.convnext.classifier[-1] = nn.Linear(self.convnext.classifier[-1].in_features, num_classes)
        layer_config = {
            "convnext_tiny": [96, 192, 384, 768],
            "convnext_small": [96, 192, 384, 768],
            "convnext_base": [128, 256, 512, 1024],
            "convnext_large": [192, 384, 768, 1536]
        }
        channels = layer_config[version]

        # KAN convolutions for each layer: b_spline, rbf, chebyshev
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 0 else (2, 2),
                                    padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for i in range(4)
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for _ in range(4)
        ])

        # Optional KAN layers
        self.kan_linear = KANLinear(in_features = self.convnext.classifier[-1].in_features, out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(96, 96 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0], channels[0] // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1], channels[1] // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2], channels[2] // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand = nn.ModuleList([
            nn.Conv2d((96 // reduce_factor[0]) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[0] // reduce_factor[1]) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[1] // reduce_factor[2]) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[2] // reduce_factor[3]) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        ## KAN normalization
        self.kan_bn1 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([96] + channels[:-1])])
        self.kan_bn2 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([96] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])

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
        if mechanism == "mult_sigmoid":
            return out * (1 + torch.sigmoid(residual))
        
        elif mechanism == "tanh":
            return out + torch.tanh(residual) * out
        
        elif mechanism == "addition":
            return out + residual
        
        elif mechanism == "se":
            se_weight = self.se_blocks[layer_index](residual)
            self._add_params([self.se_blocks[layer_index]])
            return out * se_weight + residual
        
        else:
            return None
            
    def _add_params(self, modules):
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    self.used_parameters.add(param)
    
    def forward(self, x):
        out = self.convnext.features[0](x)
        self._add_params([self.convnext.features[0]])
        
        for i in range(1, 8):
            layer = self.convnext.features[i]
            identity = out
            out = layer(out)
            self._add_params([layer])
            
            if i in [1, 3, 5] and self.mechanisms[i // 2] is not None:
                mechanism = self.mechanisms[i // 2]
                residual = self.conv_reduce[i // 2](identity)
                if not self.single_conv:
                    residual = self.kan_conv1[i // 2](residual)
                    residual = self.kan_conv2[i // 2](residual)
                    self._add_params([self.kan_conv1[i // 2], self.kan_conv2[i // 2]])
                else:
                    residual = self.kan_conv1[i // 2](residual)
                    self._add_params([self.kan_conv1[i // 2]])
                    
                residual = self.conv_expand[i // 2](residual)
                residual = self.kan_expand_bn[i // 2](residual)
                out = self.apply_mechanism(out, residual, i // 2, mechanism)
                self._add_params([self.conv_reduce[i // 2], self.conv_expand[i // 2], self.kan_expand_bn[i // 2]])
        
        out = self.convnext.avgpool(out)
        if self.fcl == "convnext":
            out = self.convnext.classifier(out)
            self._add_params([self.convnext.classifier])
        elif self.fcl == "kan":
            out = self.convnext.classifier[0](out)
            out = self.convnext.classifier[1](out)
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'convnext' or 'kan'.")

        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out