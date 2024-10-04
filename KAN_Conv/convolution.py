# Credits to: https://github.com/detkov/Convolution-From-Scratch/
# Adapted to a more vectorized operation
import torch
import numpy as np
from typing import List, Tuple, Union


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape
    h_out = np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    b = [kernel_side // 2, kernel_side// 2]
    return h_out,w_out,batch_size,n_channels

def kan_conv2d(matrix: torch.Tensor, kernel: torch.nn.Module, kernel_side: int,
               stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
               device: str = "cuda") -> torch.Tensor:
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    
    matrix_out = torch.zeros((batch_size, n_channels, h_out, w_out), device=matrix.device)
    unfold = torch.nn.Unfold(kernel_size = (kernel_side, kernel_side), dilation = dilation, padding = padding, stride = stride)

    unfolded_matrix = unfold(matrix)
    unfolded_matrix = unfolded_matrix.view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out)
    conv_groups = unfolded_matrix.permute(0, 1, 3, 2)
    
    conv_result = kernel(conv_groups.reshape(-1, kernel_side * kernel_side))
    matrix_out = conv_result.view(batch_size, n_channels, h_out, w_out)
    
    return matrix_out

def multiple_convs_kan_conv2d(matrix: torch.Tensor, kernels: List[torch.nn.Module], kernel_side: int,
                              stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1), 
                              padding: Tuple[int, int] = (0, 0), device: str = "cuda") -> torch.Tensor:
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    
    unfold = torch.nn.Unfold(kernel_size = (kernel_side, kernel_side), dilation = dilation, padding = padding, stride = stride)

    unfolded_matrix = unfold(matrix)
    unfolded_matrix = unfolded_matrix.view(batch_size, n_channels, kernel_side * kernel_side, h_out * w_out)
    conv_groups = unfolded_matrix.permute(0, 1, 3, 2)
    
    results = []
    for kernel in kernels:
        conv_result = kernel.conv.forward(conv_groups.reshape(-1, kernel_side * kernel_side))
        results.append(conv_result.view(batch_size, n_channels, h_out, w_out))
    
    matrix_out = torch.cat(results, dim = 1)
    
    return matrix_out

def add_padding(matrix: np.ndarray, 
                padding: Tuple[int, int]) -> np.ndarray:
    n, m = matrix.shape
    r, c = padding
    
    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix
    
    return padded_matrix