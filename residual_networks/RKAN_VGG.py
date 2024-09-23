import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class RKAN_VGG(nn.Module):
    def __init__(self, num_classes = 100, version = "vgg11_bn", kan_type = "chebyshev", pretrained = False, fcl = "vgg",
                 reduce_factor = 4, grid_size = 5, n_convs = 1, dataset_size = "small", single_conv = True, mechanisms = [None, None, None, None]):
        super(RKAN_VGG, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.single_conv = single_conv
        self.mechanisms = mechanisms
        self.fcl = fcl
        
        if pretrained:
            vgg = getattr(models, version)(weights = "DEFAULT")
        else:
            vgg = getattr(models, version)(weights = None)
        
        self.stem, self.stages, self.stage_indices = self._get_vgg_stages(vgg)
        self.stages = nn.ModuleList(self.stages)
        if len(self.mechanisms) != len(self.stages):
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages ({len(self.stages)}).")

        if dataset_size == "small":
            self.stem[-1] = nn.Identity()
            self.stages[0] = nn.Sequential(*list(self.stages[0].children())[:-1], nn.Identity())
        elif dataset_size == "medium":
            self.stem[-1] = nn.Identity()
        elif dataset_size == "large":
            pass
        else:
            raise ValueError(f"Invalid value for dataset_size: {dataset_size}. Choose 'small', 'medium', or 'large'.")

        self.classifier = nn.Sequential(
            nn.Linear(vgg.classifier[0].in_features, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        channels = [128, 256, 512, 512]

        # KAN convolutions for each layer
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if dataset_size == "small" and i == 0 else (2, 2),
                                    padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for i in range(len(self.stages))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for _ in range(len(self.stages))
        ])

        # Optional KAN layers
        self.kan_linear = KANLinear(in_features = vgg.classifier[-1].in_features, out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(64, 64 // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0], channels[0] // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1], channels[1] // reduce_factor, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2], channels[2] // reduce_factor, kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand = nn.ModuleList([
            nn.Conv2d((64 // reduce_factor) * n_convs, channels[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[0] // reduce_factor) * n_convs, channels[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[1] // reduce_factor) * n_convs, channels[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d((channels[2] // reduce_factor) * n_convs, channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        ## KAN normalization
        self.kan_bn1 = nn.ModuleList(nn.BatchNorm2d(ch // reduce_factor) for ch in [64] + channels[:-1])
        self.kan_bn2 = nn.ModuleList(nn.BatchNorm2d(ch // reduce_factor) for ch in [64] + channels[:-1])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])

        # Residual mechanisms
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])

    def _get_vgg_stages(self, vgg):
        features = list(vgg.features)
        maxpool_indices = [i for i, layer in enumerate(features) if isinstance(layer, nn.MaxPool2d)]
        # features[maxpool_indices[-1]] = nn.MaxPool2d(kernel_size = 2, ceil_mode = True)
        stem = nn.Sequential(*features[:maxpool_indices[0] + 1])
        stages = [nn.Sequential(*features[maxpool_indices[i] + 1:maxpool_indices[i + 1] + 1]) for i in range(len(maxpool_indices) - 1)]
        stage_indices = [maxpool_indices[0]+1] + [maxpool_indices[i]+1 for i in range(1, len(maxpool_indices))]
        return stem, stages, stage_indices
        
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
            raise ValueError(f"Invalid mechanism: {mechanism}.")
            
    def _add_params(self, modules):
        for module in modules:
            for param in module.parameters():
                if param.requires_grad:
                    self.used_parameters.add(param)
    
    def forward(self, x):
        out = self.stem(x)
        self._add_params([self.stem])
        
        for i, (stage, mechanism) in enumerate(zip(self.stages, self.mechanisms)):
            identity = out
            out = stage(out)
            self._add_params([stage])

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

        out = nn.AdaptiveAvgPool2d((7, 7))(out)
        out = torch.flatten(out, 1)
        if self.fcl == "vgg":
            out = self.classifier(out)
            self._add_params([self.classifier])
        elif self.fcl == "kan":
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'vgg' or 'kan'.")

        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out