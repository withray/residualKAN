import torch
import torch.nn as nn
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class PyramidBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(PyramidBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Handle channel mismatch with zero padding
        if out.size(1) != identity.size(1):
            batch_size, _, h, w = out.size()
            padding = torch.zeros(batch_size, out.size(1) - identity.size(1), h, w, device = out.device)
            identity = torch.cat([identity, padding], dim = 1)

        out += identity
        return out

class PyramidBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(PyramidBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Handle channel mismatch with zero padding
        if out.size(1) != identity.size(1):
            batch_size, _, h, w = out.size()
            padding = torch.zeros(batch_size, out.size(1) - identity.size(1), h, w, device = out.device)
            identity = torch.cat([identity, padding], dim = 1)

        out += identity
        return out

class RKAN_PyramidNet(nn.Module):
    def __init__(self, num_classes = 1000, version = "pyramidnet50", kan_type = "chebyshev", pretrained = False, n_convs = 1, reduce_factor = [2, 2, 2, 2],
                 mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4, alpha = None):
        super(RKAN_PyramidNet, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        pyramid_config = {
            "pyramidnet18": (PyramidBasicBlock, [2, 2, 2, 2], 64, 48),
            "pyramidnet34": (PyramidBasicBlock, [3, 4, 6, 3], 64, 48),
            "pyramidnet50": (PyramidBottleneck, [3, 4, 6, 3], 64, 200),
            "pyramidnet101": (PyramidBottleneck, [3, 4, 23, 3], 64, 270),
            "pyramidnet152": (PyramidBottleneck, [3, 8, 36, 3], 64, 270),
            "pyramidnet200": (PyramidBottleneck, [3, 24, 36, 3], 64, 240)
        }
        
        if version not in pyramid_config:
            raise ValueError(f"Unsupported version: {version}. Supported: {list(pyramid_config.keys())}.")

        block, layers, initial_channels, default_alpha = pyramid_config[version]
        if alpha is None:
            self.alpha = default_alpha
        else:
            self.alpha = alpha

        self.inplanes = initial_channels
        total_blocks = sum(layers)
        self.addrate = self.alpha / total_blocks

        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # Build layers with gradual channel increase
        self.featuremap_dim = initial_channels
        self.input_featuremap_dim = initial_channels
        
        self.layer1 = self._pyramidal_make_layer(block, layers[0], stride = 1)
        self.layer2 = self._pyramidal_make_layer(block, layers[1], stride = 2)
        self.layer3 = self._pyramidal_make_layer(block, layers[2], stride = 2)
        self.layer4 = self._pyramidal_make_layer(block, layers[3], stride = 2)

        self.bn_final = nn.BatchNorm2d(self.input_featuremap_dim)
        self.relu_final = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.input_featuremap_dim, num_classes)
        channels = self._calculate_stage_channels(layers, block)
        input_channels = [initial_channels] + channels[:-1]

        # KAN convolutions for each stage
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 0 else (2, 2), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[0], grid_size = grid_size[0])
            for i in range(4)
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1),
                                    kan_type = kan_type, spline_order = spline_order[1], grid_size = grid_size[1])
            for i in range(4)
        ])

        # Bottleneck for KAN
        if self.inv_bottleneck:
            self.conv_reduce = nn.ModuleList([nn.Conv2d(input_channels[i], input_channels[i] * self.inv_factor, kernel_size = 1, stride = 1, bias = False) for i in range(4)])
            self.conv_expand = nn.ModuleList([nn.Conv2d((input_channels[i] * self.inv_factor) * n_convs, channels[i], kernel_size = 1, stride = 1, bias = False) for i in range(4)])
        else:
            self.conv_reduce = nn.ModuleList([nn.Conv2d(input_channels[i], input_channels[i] // reduce_factor[i], kernel_size = 1, stride = 1, bias = False) for i in range(4)])
            self.conv_expand = nn.ModuleList([nn.Conv2d((input_channels[i] // reduce_factor[i]) * n_convs, channels[i], kernel_size = 1, stride = 1, bias = False) for i in range(4)])

        # KAN normalization
        bn_params = {"momentum": 0.1, "eps": 1e-5, "affine": True}
        if self.inv_bottleneck:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(input_channels[i] * self.inv_factor, **bn_params) for i in range(4)])
        else:
            self.kan_bn = nn.ModuleList([nn.BatchNorm2d(input_channels[i] // reduce_factor[i], **bn_params) for i in range(4)])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch, **bn_params) for ch in channels])

        # Activations
        self.silu = nn.SiLU()

        # Residual mechanisms
        self.gate_convs = nn.ModuleList([nn.Conv2d(ch, ch, kernel_size = 1) for ch in channels])
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])

        # Initialize weights
        self._initialize_weights()

    def _pyramidal_make_layer(self, block, block_depth, stride = 1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode = True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.expansion, int(round(temp_featuremap_dim))))
            self.featuremap_dim = temp_featuremap_dim
            
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.expansion
        return nn.Sequential(*layers)
    
    def _calculate_stage_channels(self, layers, block):
        channels = []
        temp_featuremap_dim = self.inplanes
        
        for stage_blocks in layers:
            for i in range(stage_blocks):
                temp_featuremap_dim += self.addrate
            channels.append(int(round(temp_featuremap_dim)) * block.expansion)
            
        return channels
    
    def _make_se_block(self, channels, reduction = 16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias = False),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        out = self.conv1(x)
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
                out = self.apply_mechanism(out, residual, i, mechanism)

        out = self.bn_final(out)
        out = self.relu_final(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out