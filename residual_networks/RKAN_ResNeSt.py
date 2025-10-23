import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from torchvision.models.resnet import BasicBlock, Bottleneck

class SplitAttention(nn.Module):
    def __init__(self, channels, groups = 1, radix = 2, reduction = 4, norm_layer = None):
        super(SplitAttention, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        inter_channels = max(channels * radix // reduction, 32)
        self.radix = radix
        self.groups = groups
        
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups = self.groups)
        self.bn1 = norm_layer(inter_channels)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups = self.groups)
        
        if radix > 1:
            self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        splited = []
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim = 1)
            gap = torch.stack(splited, dim = 0).sum(dim = 0)
        else:
            gap = x
        
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        
        if self.radix > 1:
            atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
            attens = torch.split(atten, rchannel // self.radix, dim = 1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            atten = torch.sigmoid(atten).view(batch, -1, 1, 1)
            out = atten * x
        
        return out

class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim = 1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class ResNeStBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation = 1, norm_layer = None, radix = 2,
                 reduction = 4, avd = False, avd_first = False):
        super(ResNeStBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        group_width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size = 1, bias = False)
        self.bn1 = norm_layer(group_width)

        self.avd = avd and stride > 1
        self.avd_first = avd_first
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding = 1)
            stride = 1 
        
        if radix >= 1:
            self.conv2 = SplitAttnConv2d(group_width, group_width, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, groups = groups,
                                         bias = False, radix = radix, reduction = reduction, norm_layer = norm_layer)
        else:
            self.conv2 = nn.Conv2d(group_width, group_width, kernel_size = 3, stride = stride, padding = dilation, dilation = dilation, groups = groups, bias = False)
            self.bn2 = norm_layer(group_width)
            
        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride
        self.radix = radix

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)
        out = self.conv2(out)

        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)
        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SplitAttnConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, radix = 2, reduction = 4, norm_layer = None):
        super(SplitAttnConv2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.radix = radix
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation, groups * radix, bias)
        self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace = True)
        self.fc1 = nn.Conv2d(channels, max(channels * radix // reduction, 32), 1, groups = groups)
        self.bn1 = norm_layer(max(channels * radix // reduction, 32))
        self.fc2 = nn.Conv2d(max(channels * radix // reduction, 32), channels * radix, 1, groups = groups)
        
        if radix > 1:
            self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        splited = []
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim = 1)
            gap = torch.stack(splited, dim = 0).sum(dim = 0)
        else:
            gap = x

        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)

        if self.radix > 1:
            atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
            attens = torch.split(atten, rchannel // self.radix, dim = 1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            atten = torch.sigmoid(atten).view(batch, -1, 1, 1)
            out = atten * x

        return out

class RKANeSt(nn.Module):
    def __init__(self, num_classes = 1000, version = "resnest50", kan_type = "chebyshev", pretrained = False, n_convs = 1, reduce_factor = [2, 2, 2, 2],
                 mechanisms = [None, None, None, "addition"], spline_order = (3, 2), grid_size = (3, 2), inv_bottleneck = False, inv_factor = 4, avd = False, avd_first = False):
        super(RKANeSt, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        self.inv_bottleneck = inv_bottleneck
        self.inv_factor = inv_factor
        self.avd = avd
        self.avd_first = avd_first

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")
        
        version_mapping = {f"resnest{i}": f"resnet{i}" for i in [50, 101]}
        backbone_version = version_mapping.get(version, version)
        
        if pretrained:
            self.resnet = getattr(models, backbone_version)(weights = "DEFAULT")
        else:
            self.resnet = getattr(models, backbone_version)(weights = None)

        block_map = {
            "resnest50": (ResNeStBottleneck, [3, 4, 6, 3]),
            "resnest101": (ResNeStBottleneck, [3, 4, 23, 3])
        }

        if version not in block_map:
            raise ValueError(f"Unsupported version: {version}. Supported: {list(block_map.keys())}.")
                
        block, layers = block_map[version]
        self.resnet.layer1 = self._replace_blocks(self.resnet.layer1, block)
        self.resnet.layer2 = self._replace_blocks(self.resnet.layer2, block)
        self.resnet.layer3 = self._replace_blocks(self.resnet.layer3, block)
        self.resnet.layer4 = self._replace_blocks(self.resnet.layer4, block)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        layer_config = {
            "resnest50": [256, 512, 1024, 2048],
            "resnest101": [256, 512, 1024, 2048]
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
                    groups = groups, base_width = base_width, dilation = dilation, norm_layer = norm_layer, avd = self.avd, avd_first = self.avd_first
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
                out = self.apply_mechanism(out, residual, i, mechanism)

        out = self.resnet.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.resnet.fc(out)
        return out