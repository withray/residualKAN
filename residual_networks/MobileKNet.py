import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class MobileKNet(nn.Module):
    def __init__(self, num_classes = 100, version = "mobilenet_v3_large", kan_type = "chebyshev", pretrained = False, fcl = "mobilenet",
                 reduce_factor = 4, grid_size = 5, n_convs = 1, small_dataset = True, single_conv = True, mechanism = True):
        super(MobileKNet, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.single_conv = single_conv
        self.mechanism = mechanism
        self.fcl = fcl
        
        if pretrained:
            self.mobilenet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.mobilenet = getattr(models, version)(weights = None)

        if small_dataset:
            if version == "mobilenet_v3_small":
                self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
                self.mobilenet.features[1].block[0][0] = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), groups = 16, bias = False)
                self.mobilenet.features[2].block[1][0] = nn.Conv2d(in_channels = 72, out_channels = 72, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), groups = 72, bias = False)
            else:
                self.mobilenet.features[0][0] = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
                self.mobilenet.features[2].block[1][0] = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), groups = 64, bias = False)

        layer_config = {
            "mobilenet_v3_large": [16, 24, 40, 80, 112, 160, 960, 1280],
            "mobilenet_v3_small": [16, 24, 40, 48, 96, 576, 1024]
        }
        channels = layer_config[version]
        self.mobilenet.classifier[-1] = nn.Linear(channels[-1], num_classes)

        # KAN convolutions for each layer: b_spline, rbf, chebyshev
        self.kan_conv1 = KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
        self.kan_conv2 = KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)

        # Optional KAN layers
        self.kan_linear = KANLinear(in_features = channels[-2], out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
        self.conv_reduce = nn.Conv2d(channels[-3], channels[-3] // reduce_factor, kernel_size = 1, stride = 1, bias = False)
        self.conv_expand = nn.Conv2d((channels[-3] // reduce_factor) * n_convs, channels[-2], kernel_size = 1, stride = 1, bias = False)

        ## KAN normalization
        self.kan_expand_bn = nn.BatchNorm2d(channels[-2])
            
    def _add_params(self, modules):
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    self.used_parameters.add(param)
    
    def forward(self, x):
        out = self.mobilenet.features[0](x)
        self._add_params([self.mobilenet.features[0]])
        
        for i in range(1, len(self.mobilenet.features)):
            if i == (len(self.mobilenet.features) - 2):
                identity = out
            layer = self.mobilenet.features[i]
            out = layer(out)
            self._add_params([layer])

            if i == (len(self.mobilenet.features) - 1):
                if self.mechanism:
                    residual = self.conv_reduce(identity)
                    if not self.single_conv:
                        residual = self.kan_conv1(residual)
                        residual = self.kan_conv2(residual)
                        self._add_params([self.kan_conv1, self.kan_conv2])
                    else:
                        residual = self.kan_conv1(residual)
                        self._add_params([self.kan_conv1])
                    residual = self.conv_expand(residual)
                    residual = self.kan_expand_bn(residual)
                    out = out + residual
                    self._add_params([self.conv_reduce, self.conv_expand, self.kan_expand_bn])

        out = self.mobilenet.avgpool(out)
        out = torch.flatten(out, 1)
        if self.fcl == "mobilenet":
            out = self.mobilenet.classifier(out)
            self._add_params([self.mobilenet.classifier])
        elif self.fcl == "kan":
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'mobilenet' or 'kan'.")

        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out