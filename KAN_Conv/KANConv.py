# Credits to: https://github.com/AntonioTepsich/Convolutional-KANs
import torch
import math
from KAN_Conv.KANLinear import KANLinear
from KAN_Conv.fastkan import FastKANLayer
from KAN_Conv.chebyshevkan import ChebyshevKANLinear
from KAN_Conv.convolution import *
import KAN_Conv.convolution as convolution

class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
            self,
            n_convs: int = 1,
            kernel_size: tuple = (3,3),
            stride: tuple = (1,1),
            padding: tuple = (1,1),
            dilation: tuple = (1,1),
            grid_size: int = 3,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation: type = torch.nn.SiLU,
            skip_activation: bool = True,
            weight_scaler: bool = True,
            grid_eps: float = 0.02,
            grid_range: tuple = (-1, 1),
            kan_type: str = "chebyshev",
            normalization = "tanh",
            use_linear: bool = True,
            use_layernorm: bool = False,
            device: str = "cpu"
        ):

        super(KAN_Convolutional_Layer, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList()
        self.n_convs = n_convs
        self.stride = stride
        self.kan_type = kan_type
        self.use_linear = use_linear

        for _ in range(n_convs):
            self.convs.append(
                KAN_Convolution(
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    grid_size = grid_size,
                    spline_order = spline_order,
                    scale_noise = scale_noise,
                    scale_base = scale_base,
                    scale_spline = scale_spline,
                    base_activation = base_activation,
                    skip_activation = skip_activation,
                    weight_scaler = weight_scaler,
                    grid_eps = grid_eps,
                    grid_range = grid_range,
                    kan_type = kan_type,
                    normalization = normalization,
                    use_linear = use_linear,
                    use_layernorm = use_layernorm,
                    device = device
                )
            )

    def forward(self, x: torch.Tensor, update_grid = False):
        if self.n_convs>1:
            return convolution.multiple_convs_kan_conv2d(x, list(self.convs), self.kernel_size[0], self.stride,self.dilation, self.padding,self.device)
        return self.convs[0].forward(x)
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 3,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation: type = torch.nn.SiLU,
            skip_activation: bool = True,
            weight_scaler: bool = True,
            grid_eps: float = 0.02,
            grid_range: tuple = (-1, 1),
            kan_type: str = "chebyshev",
            normalization = "tanh",
            use_linear: bool = True,
            use_layernorm: bool = False,
            device = "cpu"
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.device = device

        if kan_type == "rbf":
            self.conv = FastKANLayer(
                input_dim = math.prod(kernel_size),
                output_dim = 1,
                grid_min = grid_range[0],
                grid_max = grid_range[-1],
                num_grids = grid_size,
                use_base_update = use_linear,
                base_activation = base_activation,
                skip_activation = skip_activation,
                spline_weight_init_scale = scale_spline,
                use_layernorm = use_layernorm
            )

        elif kan_type in ["chebyshev", "legendre", "jacobi", "hermite"]:
            self.conv = ChebyshevKANLinear(
                in_features = math.prod(kernel_size),
                out_features = 1,
                polynomial_degree = spline_order,
                base_activation = base_activation,
                skip_activation = skip_activation,
                enable_scaler = weight_scaler,
                use_linear = use_linear,
                normalization = normalization,
                polynomial_type = kan_type,
                use_layernorm = use_layernorm
            )

        elif kan_type == "b_spline":
            self.conv = KANLinear(
                in_features = math.prod(kernel_size),
                out_features = 1,
                grid_size = grid_size,
                spline_order = spline_order,
                scale_noise = scale_noise,
                scale_base = scale_base,
                scale_spline = scale_spline,
                base_activation = base_activation,
                enable_standalone_scale_spline = weight_scaler,
                grid_eps = grid_eps,
                grid_range = grid_range,
                use_linear = use_linear,
                use_layernorm = use_layernorm
            )

        else:
            raise ValueError(f"Unsupported kan_type: {kan_type}.")

    def forward(self, x: torch.Tensor, update_grid = False):
        return convolution.kan_conv2d(x, self.conv, self.kernel_size[0], self.stride, self.dilation,self.padding, self.device)