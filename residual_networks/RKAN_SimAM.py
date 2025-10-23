import torch
import torch.nn as nn
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from torchvision.models.resnet import BasicBlock, Bottleneck

class SimAM(nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
    
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim = [2, 3], keepdim = True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim = [2, 3], keepdim = True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

class SimAMBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None, e_lambda = 1e-4):
        super(SimAMBasicBlock, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.simam = SimAM(channels = planes, e_lambda = e_lambda)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.simam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SimAMBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None, e_lambda = 1e-4):
        super(SimAMBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.simam = SimAM(channels = planes * self.expansion, e_lambda = e_lambda)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.simam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class RKAN_SimAM(nn.Module):
    def __init__(self, num_classes = 1000, version = "simam50", kan_type = "chebyshev", pretrained = False, n_convs = 1, reduce_factor = [2, 2, 2, 2],
                 mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4, shortcut = False, e_lambda = 1e-4):
        super(RKAN_SimAM, self).__init__()
        
        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor
        self.shortcut = shortcut
        self.e_lambda = e_lambda

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")
        
        version_mapping = {f"simam{i}": f"resnet{i}" for i in [18, 34, 50, 101, 152]}
        backbone_version = version_mapping.get(version, version)

        if pretrained:
            self.resnet = getattr(models, backbone_version)(weights = "DEFAULT")
        else:
            self.resnet = getattr(models, backbone_version)(weights = None)

        block_map = {
            "resnet18": (SimAMBasicBlock, [2, 2, 2, 2]),
            "resnet34": (SimAMBasicBlock, [3, 4, 6, 3]),
            "resnet50": (SimAMBottleneck, [3, 4, 6, 3]),
            "resnet101": (SimAMBottleneck, [3, 4, 23, 3]),
            "resnet152": (SimAMBottleneck, [3, 8, 36, 3])
        }

        if backbone_version not in block_map:
            raise ValueError(f"Unsupported version: {backbone_version}. Supported: {list(block_map.keys())}.")
                
        block, layers = block_map[backbone_version]
        self.resnet.layer1 = self._replace_blocks(self.resnet.layer1, block)
        self.resnet.layer2 = self._replace_blocks(self.resnet.layer2, block)
        self.resnet.layer3 = self._replace_blocks(self.resnet.layer3, block)
        self.resnet.layer4 = self._replace_blocks(self.resnet.layer4, block)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        layer_config = {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            "resnet101": [256, 512, 1024, 2048],
            "resnet152": [256, 512, 1024, 2048]
        }
        channels = layer_config[backbone_version]

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
        if self.inv_bottleneck:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch * self.inv_factor, **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        else:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i], **bn_params) for i, ch in enumerate([64] + channels[:-1])])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch, **bn_params) for ch in channels])

        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        self.gate_convs = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size = 1) for ch in channels])
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])

    def _make_se_block(self, channels, reduction = 16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias = False),
            nn.Sigmoid()
        )
    
    def _replace_blocks(self, layer, block):
        new_blocks = []
        for module in layer:
            if isinstance(module, (BasicBlock, Bottleneck)):
                inplanes = module.conv1.in_channels
                planes = module.conv1.out_channels
                stride = module.stride
                downsample = module.downsample
                groups = getattr(module, "groups", 1)
                base_width = getattr(module, "base_width", 64)
                dilation = getattr(module, "dilation", 1)
                norm_layer = type(module.bn1)
                
                new_block = block(
                    inplanes = inplanes, planes = planes, stride = stride, downsample = downsample,
                    groups = groups, base_width = base_width, dilation = dilation, norm_layer = norm_layer, e_lambda = self.e_lambda
                )
                new_blocks.append(new_block)
            else:
                new_blocks.append(module)
        return nn.Sequential(*new_blocks)

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
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)
        
        layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
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

        out = self.resnet.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.resnet.fc(out)
        return out