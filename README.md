# Residual Kolmogorov-Arnold Network (RKAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2410.05500-B31B1B)](https://arxiv.org/abs/2410.05500)

## Overview
Despite their immense success, deep neural networks (CNNs) are costly to train, while modern architectures can retain hundreds of convolutional layers in network depth. Standard convolutional operations are fundamentally limited by their linear nature along with fixed activations, where multiple layers are needed to learn complex patterns, making this approach computationally inefficient and prone to optimization difficulties. As a result, we introduce RKAN (Stage Aggregated Residual Kolmogorov-Arnold Network), which could be easily implemented into stages of traditional networks, such as ResNet. The module also integrates polynomial feature transformation that provides the expressive power of many convolutional layers through learnable, non-linear feature refinement. Our proposed RKAN module offers consistent improvements over the base models on various well-known benchmark datasets, such as CIFAR-100, Food-101, and ImageNet.

![RKAN Multi-stages](images/rkan_multistages.png)

![RKAN Stage 4 Visualization](images/rkan_stage4.png)

You can also find our paper on [arXiv](https://arxiv.org/abs/2410.05500).

Networks are trained from scratch for 200 epochs using stochastic gradient descent (SGD) with a weight decay of 0.0005 (100 epochs on ImageNet with a weight decay of 0.0001). RandAugment, CutMix with a 50\% probability, and MixUp ($\alpha = 0.2$) with a 30\% probability are used as data augmentation. RKAN blocks are added primarily to the fourth stage of the network except for ConvNeXt and Swin where the block is implemented into the second stage.

It should be noted that the RKAN module performs exceptionally well on small and datasets that are prone to overfitting. Multi-stage RKAN performs better when RKAN blocks are only implemented into the last 2 stages. More details can be found in our original paper. Performance gains for more recent architectures (ConvNeXt and Swin) are not yet observed on ImageNet.

## Results
### CIFAR-100 Results
| RKAN Model            | Top-1 Accuracy     | Base Model          | Top-1 Accuracy     |
|-----------------------|--------------------|---------------------|--------------------|
| RKAN-ResNeXt-101      | 86.15              | ResNeXt-101         | 85.28              |
| RKAN-ResNeXt-50       | 85.08              | ResNeXt-50          | 84.40              |
| RKAN-ResNet-152       | 85.40              | ResNet-152          | 84.63              |
| RKAN-ResNet-101       | 85.12              | ResNet-101          | 84.00              |
| RKAN-ResNet-50        | 84.56              | ResNet-50           | 84.12              |
| RKAN-RegNetY-32GF     | 87.03              | RegNetY-32GF        | 85.44              |
| RKAN-RegNetY-8GF      | 86.11              | RegNetY-8GF         | 84.77              |
| RKAN-RegNetY-3.2GF    | 85.46              | RegNetY-3.2GF       | 84.68              |
| RKAN-DenseNet-201     | 85.35              | DenseNet-201        | 84.28              |
| RKAN-DenseNet-169     | 84.84              | DenseNet-169        | 84.00              |
| RKAN-DenseNet-121     | 84.73              | DenseNet-121        | 84.09              |

### Food-101 Results
| RKAN Model            | Top-1 Accuracy     | Base Model          | Top-1 Accuracy     |
|-----------------------|--------------------|---------------------|--------------------|
| RKAN-ResNeXt-101      | 90.82              | ResNeXt-101         | 89.87              |
| RKAN-ResNeXt-50       | 90.00              | ResNeXt-50          | 89.20              |
| RKAN-ResNet-152       | 90.36              | ResNet-152          | 89.70              |
| RKAN-ResNet-101       | 90.09              | ResNet-101          | 89.29              |
| RKAN-ResNet-50        | 89.48              | ResNet-50           | 88.84              |
| RKAN-RegNetY-32GF     | 91.62              | RegNetY-32GF        | 90.72              |
| RKAN-RegNetY-8GF      | 91.17              | RegNetY-8GF         | 90.43              |
| RKAN-RegNetY-3.2GF    | 90.09              | RegNetY-3.2GF       | 89.54              |
| RKAN-DenseNet-201     | 89.58              | DenseNet-201        | 88.83              |
| RKAN-DenseNet-169     | 89.74              | DenseNet-169        | 89.17              |
| RKAN-DenseNet-121     | 89.43              | DenseNet-121        | 88.98              |

### Tiny ImageNet Results

<p align="left">
  <img src="images/rkan_vs_baseline.png" width="40%" />
  <img src="images/rkan_resnet.png" width="40%" />
</p>

| RKAN Model            | Top-1 Accuracy     | Base Model          | Top-1 Accuracy     |
|-----------------------|--------------------|---------------------|--------------------|
| RKAN-Wide-ResNet-101  | 77.56              | Wide-ResNet-101     | 75.46              |
| RKAN-ResNeXt-101      | 77.48              | ResNeXt-101         | 75.57              |
| RKAN-ResNeXt-50       | 75.41              | ResNeXt-50          | 73.56              |
| RKAN-ResNet-152       | 76.82              | ResNet-152          | 74.88              |
| RKAN-ResNet-101       | 76.29              | ResNet-101          | 74.51              |
| RKAN-ResNet-50        | 74.43              | ResNet-50           | 72.85              |
| RKAN-ResNet-34        | 72.03              | ResNet-34           | 70.96              |
| RKAN-RegNetY-32GF     | 77.79              | RegNetY-32GF        | 75.90              |
| RKAN-RegNetY-8GF      | 77.13              | RegNetY-8GF         | 75.58              |
| RKAN-RegNetY-3.2GF    | 76.05              | RegNetY-3.2GF       | 74.07              |
| RKAN-RegNetX-3.2GF    | 75.26              | RegNetX-3.2GF       | 73.83              |
| RKAN-DenseNet-161     | 75.79              | DenseNet-161        | 74.14              |
| RKAN-DenseNet-201     | 75.12              | DenseNet-201        | 73.10              |
| RKAN-DenseNet-169     | 74.88              | DenseNet-169        | 73.55              |
| RKAN-DenseNet-121     | 74.13              | DenseNet-121        | 72.76              |
| RKAN-ConvNeXt-T       | 72.07              | ConvNeXt-T          | 70.78              |
| RKAN-Swin-T           | 68.48              | Swin-T              | 67.05              |

### ImageNet Results
| RKAN Model            | Top-1 Accuracy     | Base Model          | Top-1 Accuracy     |
|-----------------------|--------------------|---------------------|--------------------|
| RKAN-ResNet-152       | 80.73              | ResNet-152          | 80.22              |
| RKAN-ResNet-101       | 80.09              | ResNet-101          | 79.31              |
| RKAN-ResNet-50        | 77.93              | ResNet-50           | 77.21              |
| RKAN-ResNet-34        | 74.33              | ResNet-34           | 73.72              |
| RKAN-RegNetY-8GF      | 81.38              | RegNetY-8GF         | 81.02              |
| RKAN-RegNetY-3.2GF    | 79.62              | RegNetY-3.2GF       | 79.03              |
| RKAN-RegNetX-3.2GF    | 79.11              | RegNetX-3.2GF       | 78.70              |
| RKAN-DenseNet-201     | 79.02              | DenseNet-201        | 78.41              |
| RKAN-DenseNet-169     | 78.00              | DenseNet-169        | 77.25              |
| RKAN-DenseNet-121     | 76.34              | DenseNet-121        | 75.05              |

### Citation
If you find our work useful, consider citing our paper at:
```bibtex
@article{yu2024rkan,
  title={Residual Kolmogorov-Arnold Network for Enhanced Deep Learning},
  author={Yu, Ray Congrui and Wu, Sherry and Gui, Jiang},
  journal={arXiv preprint arXiv:2410.05500},
  year={2024}
}
```


