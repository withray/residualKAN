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
            grid_size: int = 5,
            spline_order:int = 3,
            scale_noise:float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation = torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = (-1, 1),
            chebyshev_degree: int = 3,
            kan_type: str = "b_spline",
            use_linear: bool = True,
            device: str = "cpu"
        ):
        """
        Kan Convolutional Layer with multiple convolutions
        
        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base
            scale_spline (float): Scale of the spline
            base_activation (torch.nn.Module): Activation function
            grid_eps (float): Epsilon of the grid
            grid_range (tuple): Range of the grid
            device (str): Device to use
        """


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
        self.chebyshev_degree = chebyshev_degree
        self.kan_type = kan_type
        self.use_linear = use_linear

        # Create n_convs KAN_Convolution objects
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
                    grid_eps = grid_eps,
                    grid_range = grid_range,
                    chebyshev_degree = chebyshev_degree,
                    kan_type = kan_type,
                    use_linear = self.use_linear,
                    device = device
                )
            )

    def forward(self, x: torch.Tensor, update_grid = False):
        # If there are multiple convolutions, apply them all
        if self.n_convs>1:
            return convolution.multiple_convs_kan_conv2d(x, list(self.convs), self.kernel_size[0], self.stride,self.dilation, self.padding,self.device)
        
        # If there is only one convolution, apply it
        return self.convs[0].forward(x)
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = (-1, 1),
            chebyshev_degree: int = 3,
            kan_type: str = "b_spline",
            use_linear: bool = True,
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
                grid_min = -2,
                grid_max = 2,
                num_grids = grid_size,
                use_base_update = use_linear,
                base_activation = base_activation,
                use_layernorm = True,
                spline_weight_init_scale = scale_spline,
            )

        elif kan_type == "chebyshev":
            self.conv = ChebyshevKANLinear(
                in_features = math.prod(kernel_size),
                out_features = 1,
                chebyshev_degree = chebyshev_degree,
                base_activation = base_activation,
                use_linear = use_linear,
                normalization = "tanh",
                use_layernorm = False,
                clip_min = -5.0,
                clip_max = 5.0,
                use_clip = False,
                eps = 1e-8
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
                grid_eps = grid_eps,
                grid_range = grid_range,
                use_linear = use_linear,
                use_layernorm = False
            )

        else:
            raise ValueError(f"Unsupported kan_type: {kan_type}. Choose from 'rbf', 'chebyshev', 'b_spline'.")

    def forward(self, x: torch.Tensor, update_grid = False):
        return convolution.kan_conv2d(x, self.conv, self.kernel_size[0], self.stride, self.dilation,self.padding, self.device)  