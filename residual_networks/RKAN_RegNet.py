import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class RKAN_RegNet(nn.Module):
    def __init__(self, num_classes = 100, version = "regnet_y_400mf", kan_type = "chebyshev", pretrained = False, fcl = "regnet",
                 reduce_factor = 4, grid_size = 5, n_convs = 1, dataset_size = "small", single_conv = True, mechanisms = [None, None, None, None], scaling = False):
        super(RKAN_RegNet, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.single_conv = single_conv
        self.mechanisms = mechanisms
        self.fcl = fcl
        self.scaling = scaling

        if self.scaling:
            self.scaling_factor = nn.Parameter(torch.ones(1))

        if pretrained:
            self.regnet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.regnet = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        if dataset_size == "small":
            self.regnet.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace = True)
            )
            self.regnet.trunk_output[0][0].proj[0].stride = (1, 1)
            self.regnet.trunk_output[0][0].f.b[0].stride = (1, 1)
        elif dataset_size == "medium":
            self.regnet.trunk_output[0][0].proj[0].stride = (1, 1)
            self.regnet.trunk_output[0][0].f.b[0].stride = (1, 1)
        elif dataset_size == "large":
            pass
        else:
            raise ValueError(f"Invalid value for dataset_size: {dataset_size}. Choose 'small', 'medium', or 'large'.")

        self.regnet.fc = nn.Linear(self.regnet.fc.in_features, num_classes)
        layer_config = {
            "regnet_y_400mf": [48, 104, 208, 440],
            "regnet_y_800mf": [64, 144, 320, 784],
            "regnet_y_1_6gf": [48, 120, 336, 888],
            "regnet_y_3_2gf": [72, 216, 576, 1512]
        }
        channels = layer_config[version]


        # KAN convolutions for each layer
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if dataset_size == "small" and i == 0 else (2, 2),
                                    padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for i in range(4)
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for _ in range(4)
        ])

        # Optional KAN layers
        self.kan_linear = KANLinear(in_features = self.regnet.fc.in_features, out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(32, 32 // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0], channels[0] // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1], channels[1] // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2], channels[2] // reduce_factor, kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand = nn.ModuleList([
            nn.Conv2d(32 // reduce_factor, channels[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0] // reduce_factor, channels[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1] // reduce_factor, channels[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2] // reduce_factor, channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        ## KAN normalization
        self.kan_bn1 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor) for ch in [32] + channels[:-1]])
        self.kan_bn2 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor) for ch in [32] + channels[:-1]])
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
        if self.scaling:
            residual = residual * self.scaling_factor
            
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
            raise ValueError(f"Invalid mechanism: {mechanism}.")
        
    def _add_params(self, modules):
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    self.used_parameters.add(param)
    
    def forward(self, x):
        out = self.regnet.stem(x)
        self._add_params([self.regnet.stem])
        for i, (block, mechanism) in enumerate(zip(self.regnet.trunk_output, self.mechanisms)):
            identity = out
            out = block(out)
            self._add_params([block])

            if mechanism is not None:
                residual = self.conv_reduce[i](identity)
                if not self.single_conv:
                    residual = self.kan_conv1[i](residual)
                    residual = self.kan_conv2[i](residual)
                    self._add_params([self.kan_conv1[i], self.kan_conv2[i]])
                else:
                    residual = self.kan_conv1[i](residual)
                    self._add_params([self.kan_conv1[i]])
                residual = self.conv_expand[i](residual)
                residual = self.kan_expand_bn[i](residual)
                out = self.apply_mechanism(out, residual, i, mechanism)
                self._add_params([self.conv_reduce[i], self.conv_expand[i], self.kan_expand_bn[i]])

        out = self.regnet.avgpool(out)
        out = torch.flatten(out, 1)
        if self.fcl == "regnet":
            out = self.regnet.fc(out)
            self._add_params([self.regnet.fc])
        elif self.fcl == "kan":
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'regnet' or 'kan'.")
        
        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out