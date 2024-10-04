import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class RKAN_ResNeXt(nn.Module):
    def __init__(self, num_classes = 100, version = "resnext50_32x4d", kan_type = "chebyshev", pretrained = False, main_conv = "resnext", fcl = "resnext", log_norms = False,
                 reduce_factor = [4, 4, 4, 4], grid_size = 5, n_convs = 1, dataset_size = "small", single_conv = True, mechanisms = [None, None, None, None], scaling = False):
        super(RKAN_ResNeXt, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.single_conv = single_conv
        self.mechanisms = mechanisms
        self.fcl = fcl
        self.main_conv = main_conv
        self.log_norms = log_norms
        self.scaling = scaling

        if self.scaling:
            self.scaling_factor = nn.Parameter(torch.ones(1))
        
        if pretrained:
            self.resnext = getattr(models, version)(weights = "DEFAULT")
        else:
            self.resnext = getattr(models, version)(weights = None)

        if log_norms:
            self.base_norms = []
            self.residual_norms = []
            self.combined_norms = []

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        if dataset_size == "small":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size)
            self.resnext.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.resnext.maxpool = nn.Identity()
        elif dataset_size == "medium":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), grid_size = grid_size)
            self.resnext.maxpool = nn.Identity()
        elif dataset_size == "large":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), grid_size = grid_size)
        else:
            raise ValueError(f"Invalid value for dataset_size: {dataset_size}. Choose 'small', 'medium', or 'large'.")

        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)
        layer_config = {
            "resnext50_32x4d": [256, 512, 1024, 2048],
            "resnext101_32x8d": [256, 512, 1024, 2048]
        }
        channels = layer_config[version]

        # KAN convolutions for each layer
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
        self.conv_expand_main = nn.Conv2d(3, 64, kernel_size = 1, stride = 1, bias = False)
        self.kan_linear = KANLinear(in_features = self.resnext.fc.in_features, out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
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

        ## KAN normalization
        self.kan_bn1 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_bn2 = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate([64] + channels[:-1])])
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

    def reset_norms(self):
        self.base_norms = []
        self.residual_norms = []
        self.combined_norms = []
    
    def forward(self, x):
        if self.main_conv == "resnext":
            out = self.resnext.conv1(x)
            out = self.resnext.bn1(out)
            out = self.resnext.relu(out)
            out = self.resnext.maxpool(out)
            self._add_params([self.resnext.conv1, self.resnext.bn1])
        elif self.main_conv == "kan":
            out = self.conv_expand_main(self.kan_conv_main(x))
            out = self.resnext.bn1(out)
            out = self.resnext.maxpool(out)
            self._add_params([self.kan_conv_main, self.conv_expand_main, self.resnext.bn1])
        else:
            raise ValueError(f"Invalid value for main_conv: {self.main_conv}. Choose 'resnext' or 'kan'.")
        
        layers = [self.resnext.layer1, self.resnext.layer2, self.resnext.layer3, self.resnext.layer4]
        for i, (layer, mechanism) in enumerate(zip(layers, self.mechanisms)):
            identity = out
            out = layer(out)
            self._add_params([layer])

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
                if self.log_norms:
                    self.base_norms.append(torch.norm(out).item())
                    self.residual_norms.append(torch.norm(residual).item())
                out = self.apply_mechanism(out, residual, i, mechanism)
                if self.log_norms:
                    self.combined_norms.append(torch.norm(out).item())
                self._add_params([self.conv_reduce[i], self.conv_expand[i], self.kan_expand_bn[i]])

        out = self.resnext.avgpool(out)
        out = torch.flatten(out, 1)
        if self.fcl == "resnext":
            out = self.resnext.fc(out)
            self._add_params([self.resnext.fc])
        elif self.fcl == "kan":
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'resnext' or 'kan'.")

        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out