import torch.nn as nn
import torchvision.models as models
from KAN_Conv.KANConv import KAN_Convolutional_Layer

class RKAN_Swin(nn.Module):
    def __init__(self, num_classes = 1000, version = "swin_s", kan_type = "chebyshev", pretrained = False, reduce_factor = [2, 2, 2, 2],
                 n_convs = 1, mechanisms = [None, "addition", None, None], normalization = "bn"):
        super(RKAN_Swin, self).__init__()

        self.mechanisms = mechanisms
        self.reduce_factor = reduce_factor
        if normalization not in ["bn", "ln"]:
            raise ValueError(f"Normalization type '{normalization}' not supported.")
        self.normalization = normalization
        
        if pretrained:
            self.swin = getattr(models, version)(weights = "DEFAULT")
        else:
            self.swin = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)
        layer_config = {
            "swin_t": [96, 192, 384, 768],
            "swin_s": [96, 192, 384, 768],
            "swin_b": [128, 256, 512, 1024]
        }
        channels = layer_config[version]

        # KAN convolutions for each layer
        self.kan_conv1 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), kan_type = kan_type, spline_order = 3)
            for i in range(len(channels))
        ])

        self.kan_conv2 = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), kan_type = kan_type, spline_order = 2)
            for i in range(len(channels))
        ])

        # Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([nn.Conv2d(ch, ch // reduce_factor[i], kernel_size = 1, stride = 1, bias = False) for i, ch in enumerate(channels)])
        self.conv_expand = nn.ModuleList([nn.Conv2d((ch // reduce_factor[i]) * n_convs, ch, kernel_size = 1, stride = 1, bias = False)for i, ch in enumerate(channels)])

        # KAN normalization
        self.kan_bn = nn.ModuleList([nn.BatchNorm2d(ch // reduce_factor[i]) for i, ch in enumerate(channels)])
        self.kan_expand_bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in channels])
        self.kan_ln = nn.ModuleList([nn.LayerNorm(channels[i] // reduce_factor[i]) for i in range(len(channels))])
        self.kan_expand_ln = nn.ModuleList([nn.LayerNorm(ch) for ch in channels])
        
        # Activations
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        
        # Residual mechanisms
        self.se_blocks = nn.ModuleList([self._make_se_block(ch, reduction = 16) for ch in channels])

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
            se_weight = self.se_blocks[layer_index](residual.permute(0, 3, 1, 2))
            return out + residual * se_weight.permute(0, 2, 3, 1)
        
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}.")

    def forward(self, x):
        out = self.swin.features[0](x)
        stage_indices = [1, 3, 5, 7]
        
        for idx, i in enumerate(stage_indices):
            identity = out
            out = self.swin.features[i](out)

            if self.mechanisms[idx] is not None:
                identity = identity.permute(0, 3, 1, 2)
                residual = self.conv_reduce[idx](identity)
                residual = self.silu(residual)
                
                residual = self.kan_conv1[idx](residual)
                if self.normalization == "bn":
                    residual = self.kan_bn[idx](residual)
                else:
                    residual = residual.permute(0, 2, 3, 1)
                    residual = self.kan_ln[idx](residual)
                    residual = residual.permute(0, 3, 1, 2)
                    
                residual = self.conv_expand[idx](residual)
                residual = self.silu(residual)

                residual = self.kan_conv2[idx](residual)
                if self.normalization == "bn":
                    residual = self.kan_expand_bn[idx](residual)
                    residual = residual.permute(0, 2, 3, 1)
                else:
                    residual = residual.permute(0, 2, 3, 1)
                    residual = self.kan_expand_ln[idx](residual)
                out = self.apply_mechanism(out, residual, idx, self.mechanisms[idx])
            
            if i != stage_indices[-1]:
                out = self.swin.features[i + 1](out)

        out = self.swin.norm(out)
        out = self.swin.permute(out)
        out = self.swin.avgpool(out)
        out = self.swin.flatten(out)
        out = self.swin.head(out)
        return out