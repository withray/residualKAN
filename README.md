# Residual Kolmogorov-Arnold Network (RKAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2410.05500-B31B1B)](https://arxiv.org/abs/2410.05500)

## Overview
Despite their immense success, deep convolutional neural networks (CNNs) can be difficult to optimize and costly to train due to hundreds of layers within the network depth. Conventional convolutional operations are fundamentally limited by their linear nature along with fixed activations, where many layers are needed to learn meaningful patterns in data. Because of the sheer size of these networks, this approach is simply computationally inefficient, and poses overfitting or gradient vanishing risks, especially in small datasets. As a result, we introduce a ``plug-in" module, called Residual Kolmogorov-Arnold Network (RKAN). Our module is highly compact, so it can be easily added into any stage (level) of traditional deep networks, where it learns to integrate supportive polynomial feature transformations to existing convolutional frameworks. RKAN offers consistent improvements over baseline models in different vision tasks and widely tested benchmarks, accomplishing cutting-edge performance on small-scale datasets.

![RKAN Multi-stages](images/rkan_multistages.png)

![RKAN Stage 4 Visualization](images/rkan_stage4.png)

You can also find our paper on [arXiv](https://arxiv.org/abs/2410.05500).

Networks are trained from scratch for 200 epochs using stochastic gradient descent (SGD) with a weight decay of 0.0005 (100 epochs on ImageNet with a weight decay of 0.0001). RandAugment, CutMix with a 50\% probability, and MixUp ($\alpha = 0.2$) with a 30\% probability are used as data augmentation. RKAN blocks are added to the last stage of the network. ResNet is set to the default backbone, where RKAN-ResNet-101 is shortened as RKANet-101.

It should be noted that the RKAN module performs better on small datasets that are prone to overfitting than larger datasets, such as ImageNet or COCO. Multi-stage RKAN performs better when RKAN blocks are only implemented into the last 2 stages. More details can be found in our original paper.

## Usage
All necessary code is included in the repository to run RKAN with different backbone architectures on different datasets.
1. Clone the repository or download the ZIP file
2. Run the `training.ipynb` notebook 
3. Key configuration parameters:
   ```python
   # Select dataset
   dataset = "cifar_100"  # Options: cifar_100, cifar_10, svhn, tiny_imagenet, food_101, caltech_256, imagenet_1k
   
   # Select model
   model_name = "resnet50"  # See model_configs for all supported models
   
   # RKAN configuration
   reduce_factor = [2, 2, 2, 2]  # Reduce factors for each stage
   mechanisms = ["addition", "addition", "addition", "addition"]  # Aggregation mechanism for each stage, input None to remove RKAN from the stage (added only to stage 4 by default)
   kan_type = "chebyshev"  # Type of KAN convolutions, including chebyshev, rbf, b_spline, jacobi, hermite, etc.
   

## Results
### CIFAR-100 Results
| RKAN Model            | Top-1 Accuracy     | Base Model          | Top-1 Accuracy     |
|-----------------------|--------------------|---------------------|--------------------|
| RKAN-PyramidNet-200   | 86.35              | PyramidNet-200      | 85.62              |
| RKANeXt-101           | 86.15              | ResNeXt-101         | 85.28              |
| RKANeSt-101           | 85.52              | ResNeSt-101         | 84.47              |
| RKAN-SENet-101        | 85.39              | SENet-101           | 84.36              |
| RKANet-152            | 85.40              | ResNet-152          | 84.63              |
| RKANet-101-D          | 86.07              | ResNet-101-D        | 85.09              |
| RKANet-101            | 85.12              | ResNet-101          | 84.00              |
| RKAN-RegNetY-32GF     | 87.03              | RegNetY-32GF        | 85.44              |
| RKAN-RegNetY-8GF      | 86.11              | RegNetY-8GF         | 84.77              |
| RKAN-RegNetY-3.2GF    | 85.46              | RegNetY-3.2GF       | 84.68              |
| RKAN-DenseNet-201     | 85.35              | DenseNet-201        | 84.28              |
| RKAN-DenseNet-169     | 84.84              | DenseNet-169        | 84.00              |
| RKAN-DenseNet-121     | 84.73              | DenseNet-121        | 84.09              |

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


