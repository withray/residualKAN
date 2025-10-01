import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from residual_networks.RKAN_ResNet import RKANet
from residual_networks.RKAN_DenseNet import RKAN_DenseNet

class RKAN_RCNN(nn.Module):
    def __init__(self, num_classes, backbone_name = "resnet50", kan_type = "chebyshev", detector_type = "faster", pretrained = False, n_convs = 1,
                 reduce_factor = [2, 2, 2, 2], mechanisms = [None, None, None, "addition"], input_size = 600, shortcut = False):
        super(RKAN_RCNN, self).__init__()
        
        if backbone_name.startswith("resnet"):
            if backbone_name == "resnet18":
                base_model = models.resnet18(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "resnet34":
                base_model = models.resnet34(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "resnet50":
                base_model = models.resnet50(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "resnet101":
                base_model = models.resnet101(weights = "DEFAULT" if pretrained else None)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            setattr(base_model, "fc", nn.Identity())
            setattr(base_model, "avgpool", nn.Identity())
            self.detector_type = detector_type
            self.rkan_components = RKANet(num_classes = 1000, version = backbone_name, kan_type = kan_type, pretrained = False, n_convs = n_convs,
                                            reduce_factor = reduce_factor, mechanisms = mechanisms, shortcut = shortcut)
            
            # Wrapper layers that include KAN
            self.stage1_kan = self._make_resnet_kan_layer(base_model.layer1, 0)
            self.stage2_kan = self._make_resnet_kan_layer(base_model.layer2, 1)
            self.stage3_kan = self._make_resnet_kan_layer(base_model.layer3, 2)
            self.stage4_kan = self._make_resnet_kan_layer(base_model.layer4, 3)
            
            # Replace base model layers with KAN
            setattr(base_model, "layer1", self.stage1_kan)
            setattr(base_model, "layer2", self.stage2_kan)
            setattr(base_model, "layer3", self.stage3_kan)
            setattr(base_model, "layer4", self.stage4_kan)
            
            if backbone_name in ["resnet50", "resnet101", "resnet152"]:
                in_channels_list = [256, 512, 1024, 2048]
            else:
                in_channels_list = [64, 128, 256, 512]

            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            backbone = base_model

        elif backbone_name.startswith("densenet"):
            if backbone_name == "densenet121":
                base_model = models.densenet121(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "densenet169":
                base_model = models.densenet169(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "densenet201":
                base_model = models.densenet201(weights = "DEFAULT" if pretrained else None)
            elif backbone_name == "densenet161":
                base_model = models.densenet161(weights = "DEFAULT" if pretrained else None)
            else:
                raise ValueError(f"Unsupported backbone: {backbone_name}")
            
            setattr(base_model, "classifier", nn.Identity())
            self.detector_type = detector_type
            self.rkan_components = RKAN_DenseNet(num_classes = 1000, version = backbone_name, kan_type = kan_type, pretrained = False, n_convs = n_convs,
                                                 reduce_factor = reduce_factor, mechanisms = mechanisms)
            
            self.stage1_kan = self._make_densenet_kan_layer(base_model.features.denseblock1, base_model.features.transition1, 0)
            self.stage2_kan = self._make_densenet_kan_layer(base_model.features.denseblock2, base_model.features.transition2, 1)
            self.stage3_kan = self._make_densenet_kan_layer(base_model.features.denseblock3, base_model.features.transition3, 2)
            self.stage4_kan = self._make_densenet_kan_layer(base_model.features.denseblock4, base_model.features.norm5, 3)
            
            setattr(base_model.features, "denseblock1", self.stage1_kan)
            setattr(base_model.features, "denseblock2", self.stage2_kan)
            setattr(base_model.features, "denseblock3", self.stage3_kan)
            setattr(base_model.features, "denseblock4", self.stage4_kan)
            
            # Remove transition layers and norm5
            setattr(base_model.features, "transition1", nn.Identity())
            setattr(base_model.features, "transition2", nn.Identity())
            setattr(base_model.features, "transition3", nn.Identity())
            setattr(base_model.features, "norm5", nn.Identity())
            
            densenet_channels = {
                "densenet121": [128, 256, 512, 1024],
                "densenet169": [128, 256, 640, 1664],
                "densenet201": [128, 256, 896, 1920], 
                "densenet161": [192, 384, 1056, 2208]
            }

            in_channels_list = densenet_channels[backbone_name]

            class DenseNetBackboneWrapper(nn.Module):
                def __init__(self, densenet_model):
                    super().__init__()
                    
                    self.conv0 = densenet_model.features.conv0
                    self.norm0 = densenet_model.features.norm0
                    self.relu0 = densenet_model.features.relu0
                    self.pool0 = densenet_model.features.pool0
                    
                    self.stage1 = densenet_model.features.denseblock1
                    self.stage2 = densenet_model.features.denseblock2 
                    self.stage3 = densenet_model.features.denseblock3
                    self.stage4 = densenet_model.features.denseblock4
                    
                def forward(self, x):
                    x = self.conv0(x)
                    x = self.norm0(x) 
                    x = self.relu0(x)
                    x = self.pool0(x)

                    x = self.stage1(x)
                    x = self.stage2(x)
                    x = self.stage3(x)
                    x = self.stage4(x)
                    return x
                
            backbone = DenseNetBackboneWrapper(base_model)
            return_layers = {"stage1": "0", "stage2": "1", "stage3": "2", "stage4": "3"}

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_with_fpn = BackboneWithFPN(backbone, return_layers = return_layers, in_channels_list = in_channels_list, out_channels = 256)
        anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512),) * 5, aspect_ratios = ((0.5, 1.0, 2.0),) * 5)
        if detector_type == "faster":
            self.model = FasterRCNN(self.backbone_with_fpn, num_classes = num_classes, rpn_anchor_generator = anchor_generator, min_size = input_size, max_size = int(input_size * 1.666))
        elif detector_type == "mask":
            self.model = MaskRCNN(self.backbone_with_fpn, num_classes = num_classes, rpn_anchor_generator = anchor_generator, min_size = input_size, max_size = int(input_size * 1.666))
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}.")
        
    def _make_resnet_kan_layer(self, layer, layer_idx):
        rkan = self.rkan_components
        
        class ResNetKANLayer(nn.Module):
            def __init__(self, base_layer, idx):
                super().__init__()
                self.base_layer = base_layer
                self.idx = idx
                
            def forward(self, x):
                identity = x
                out = self.base_layer(x)
                
                if self.idx < len(rkan.mechanisms) and rkan.mechanisms[self.idx] is not None:
                    mechanism = rkan.mechanisms[self.idx]
                    
                    residual = rkan.conv_reduce[self.idx](identity)
                    residual = rkan.silu(residual)

                    residual = rkan.kan_conv1[self.idx](residual)
                    residual = rkan.kan_bn[self.idx](residual)

                    residual = rkan.conv_expand[self.idx](residual)
                    residual = rkan.silu(residual)
                    
                    if self.idx == len(rkan.mechanisms) - 1:
                        residual = rkan.kan_conv2[self.idx](residual)
                    residual = rkan.kan_expand_bn[self.idx](residual)
                    
                    if rkan.shortcut:
                        shortcut = rkan.conv_shortcut[self.idx](identity)
                        shortcut = rkan.shortcut_bn[self.idx](shortcut)
                        residual = residual + shortcut
                    out = rkan.apply_mechanism(out, residual, self.idx, mechanism)
                return out
            
        return ResNetKANLayer(layer, layer_idx)
    
    def _make_densenet_kan_layer(self, denseblock, post_transition, layer_idx):
        rkan = self.rkan_components
        
        class DenseNetKANLayer(nn.Module):
            def __init__(self, dense_block, post_transition, idx):
                super().__init__()
                self.dense_block = dense_block
                self.post_transition = post_transition
                self.idx = idx
                
            def forward(self, x):
                identity = x
                out = self.dense_block(x)
                
                # Apply post-processing
                out = self.post_transition(out)
                
                if self.idx < len(rkan.mechanisms) and rkan.mechanisms[self.idx] is not None:
                    mechanism = rkan.mechanisms[self.idx]
                    
                    residual = rkan.conv_reduce[self.idx](identity)
                    residual = rkan.silu(residual)
                    
                    residual = rkan.kan_conv1[self.idx](residual)
                    residual = rkan.kan_bn[self.idx](residual)
                    
                    residual = rkan.conv_expand[self.idx](residual)
                    residual = rkan.silu(residual)
                    
                    if self.idx == len(rkan.mechanisms) - 1:
                        residual = rkan.kan_conv2[self.idx](residual)
                    residual = rkan.kan_expand_bn[self.idx](residual)
                    out = rkan.apply_mechanism(out, residual, self.idx, mechanism)
                return out
            
        return DenseNetKANLayer(denseblock, post_transition, layer_idx)
        
    def forward(self, images, targets = None):
        return self.model(images, targets)
    
    def train(self, mode = True):
        super().train(mode)
        self.model.train(mode)
        return self