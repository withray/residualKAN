import torch.nn as nn
import torchvision.models as models
import torch
from KAN_Conv.KANConv import KAN_Convolutional_Layer
from KAN_Conv.KANLinear import KANLinear

class RKAN_DenseNet(nn.Module):
    def __init__(self, num_classes = 100, version = "densenet121", kan_type = "chebyshev", pretrained = False, main_conv = "densenet", fcl = "densenet",
                 reduce_factor = [4, 4, 4, 4], grid_size = 5, n_convs = 1, dataset_size = "small", pre_transition = False, mechanisms = [None, None, None, None]):
        super(RKAN_DenseNet, self).__init__()

        self.used_parameters = set()
        self.printed_params = False
        self.pre_transition = pre_transition
        self.mechanisms = mechanisms
        self.fcl = fcl
        self.main_conv = main_conv
        
        if pretrained:
            self.densenet = getattr(models, version)(weights = "DEFAULT")
        else:
            self.densenet = getattr(models, version)(weights = None)

        if len(self.mechanisms) != 4:
            raise ValueError(f"Length of mechanisms ({len(self.mechanisms)}) must match the number of stages (4).")

        if dataset_size == "small":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size)
            self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
            self.densenet.features.pool0 = nn.Identity()
        elif dataset_size == "medium":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), grid_size = grid_size)
            self.densenet.features.pool0 = nn.Identity()
        elif dataset_size == "large":
            self.kan_conv_main = KAN_Convolutional_Layer(n_convs = 1, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3), grid_size = grid_size)
        else:
            raise ValueError(f"Invalid value for dataset_size: {dataset_size}. Choose 'small', 'medium', or 'large'.")
        
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        layer_config = {
            "densenet121": [256, 512, 1024, 1024],
            "densenet169": [256, 512, 1280, 1664],
            "densenet201": [256, 512, 1792, 1920]
        }
        channels = layer_config[version]

        # KAN convolutions for each layer
        self.kan_conv_pre = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for _ in range(4)
        ])

        self.kan_conv_post = nn.ModuleList([
            KAN_Convolutional_Layer(n_convs = n_convs, kernel_size = (3, 3), stride = (1, 1) if i == 3 else (2, 2),
                                    padding = (1, 1), grid_size = grid_size, kan_type = kan_type)
            for i in range(4)
        ])

        # Optional KAN layers
        self.conv_expand_main = nn.Conv2d(3, 64, kernel_size = 1, stride = 1, bias = False)
        self.kan_linear = KANLinear(in_features = self.densenet.classifier.in_features, out_features = num_classes, grid_size = grid_size)

        ## Bottleneck for KAN
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(64, 64 // reduce_factor[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0] // 2, channels[0] // 2 // reduce_factor[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1] // 2, channels[1] // 2 // reduce_factor[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2] // 2, channels[2] // 2 // reduce_factor[3], kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand_pre = nn.ModuleList([
            nn.Conv2d(64 // reduce_factor[0], channels[0], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0] // 2 // reduce_factor[1], channels[1], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1] // 2 // reduce_factor[2], channels[2], kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2] // 2 // reduce_factor[3], channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        self.conv_expand_post = nn.ModuleList([
            nn.Conv2d(64 // reduce_factor[0], channels[0] // 2, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[0] // 2 // reduce_factor[1], channels[1] // 2, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[1] // 2 // reduce_factor[2], channels[2] // 2, kernel_size = 1, stride = 1, bias = False),
            nn.Conv2d(channels[2] // 2 // reduce_factor[3], channels[3], kernel_size = 1, stride = 1, bias = False)
        ])

        ## KAN normalization
        self.kan_bn = nn.ModuleList([nn.BatchNorm2d(channels[0] // 2), nn.BatchNorm2d(channels[1] // 2), nn.BatchNorm2d(channels[2] // 2), nn.BatchNorm2d(channels[3])])

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
        if self.main_conv == "densenet":
            out = self.densenet.features.conv0(x)
            out = self.densenet.features.norm0(out)
            out = self.densenet.features.relu0(out)
            out = self.densenet.features.pool0(out)
            self._add_params([self.densenet.features.conv0, self.densenet.features.norm0])
        elif self.main_conv == "kan":
            out = self.conv_expand_main(self.kan_conv_main(x))
            out = self.densenet.features.norm0(out)
            out = self.densenet.features.pool0(out)
            self._add_params([self.kan_conv_main, self.conv_expand_main, self.densenet.features.norm0])
        else:
            raise ValueError(f"Invalid value for main_conv: {self.main_conv}. Choose 'densenet' or 'kan'.")
        
        dense_blocks = [self.densenet.features.denseblock1, self.densenet.features.denseblock2, self.densenet.features.denseblock3, self.densenet.features.denseblock4]
        transition_layers = [self.densenet.features.transition1, self.densenet.features.transition2, self.densenet.features.transition3]
        
        for i, (block, mechanism) in enumerate(zip(dense_blocks, self.mechanisms)):
            identity = out
            out = block(out)
            self._add_params([block])
            
            # Apply norm5 and ReLU before adding residual for the last denseblock if post-transition
            if not self.pre_transition and i == len(dense_blocks) - 1:
                out = self.densenet.features.norm5(out)
                out = torch.relu(out)
                self._add_params([self.densenet.features.norm5])

            # Pre-transition mechanism without batch normalization
            if self.pre_transition and mechanism is not None:
                residual = self.conv_reduce[i](identity)
                residual = self.kan_conv_pre[i](residual)
                residual = self.conv_expand_pre[i](residual)
                out = self.apply_mechanism(out, residual, i, mechanism)
                self._add_params([self.conv_reduce[i], self.kan_conv_pre[i], self.conv_expand_pre[i]])

            # Apply norm5 and ReLU after adding residual to the output of the last dense block if pre-transition
            if self.pre_transition and i == len(dense_blocks) - 1:
                out = self.densenet.features.norm5(out)
                out = torch.relu(out)
                self._add_params([self.densenet.features.norm5])

            if i < len(transition_layers):
                out = transition_layers[i](out)
                self._add_params([transition_layers[i]])

            # Post-transition mechanism
            if not self.pre_transition and mechanism is not None:
                residual = self.conv_reduce[i](identity)
                residual = self.kan_conv_post[i](residual)
                residual = self.conv_expand_post[i](residual)
                residual = self.kan_bn[i](residual)
                out = self.apply_mechanism(out, residual, i, mechanism)
                self._add_params([self.conv_reduce[i], self.kan_conv_post[i], self.conv_expand_post[i], self.kan_bn[i]])

        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        if self.fcl == "densenet":
            out = self.densenet.classifier(out)
            self._add_params([self.densenet.classifier])
        elif self.fcl == "kan":
            out = self.kan_linear(out)
            self._add_params([self.kan_linear])
        else:
            raise ValueError(f"Invalid value for fcl: {self.fcl}. Choose 'densenet' or 'kan'.")
        
        if not self.printed_params:
            total_params = sum(p.numel() for p in self.used_parameters)
            print(f"Total Parameters: {total_params / 1e6:.4f}M")
            self.printed_params = True
        return out