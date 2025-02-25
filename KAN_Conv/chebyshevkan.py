import torch
import math

class ChebyshevKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        chebyshev_degree = 3,
        enable_chebyshev_scaler = False,
        base_activation = torch.nn.SiLU,
        use_linear = True,
        skip_activation = True,
        normalization = "tanh",
        use_legendre = False,
        use_layernorm = False,
    ):
        super(ChebyshevKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chebyshev_degree = chebyshev_degree
        self.use_linear = use_linear
        self.skip_activation = skip_activation
        self.normalization = normalization
        self.use_legendre = use_legendre
        self.use_layernorm = use_layernorm

        if self.use_linear:
            self.base_linear = torch.nn.Linear(in_features, out_features, bias = False)
        self.chebyshev_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, chebyshev_degree + 1))
        if enable_chebyshev_scaler:
            self.chebyshev_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.enable_chebyshev_scaler = enable_chebyshev_scaler
        self.base_activation = base_activation()

        self.reset_parameters()
        if self.use_layernorm:
            self.layernorm = torch.nn.LayerNorm(out_features)

    def reset_parameters(self):
        with torch.no_grad():
            std_dev = 1.0 / math.sqrt(self.in_features * (self.chebyshev_degree + 1))
            torch.nn.init.normal_(self.chebyshev_weight, mean = 0.0, std = std_dev)
            if self.use_linear:
                torch.nn.init.kaiming_normal_(self.base_linear.weight, nonlinearity = "relu")
            if self.enable_chebyshev_scaler:
                torch.nn.init.constant_(self.chebyshev_scaler, 1.0)

    def chebyshev_polynomials(self, x):
        T = [torch.ones_like(x), x]
        for _ in range(2, self.chebyshev_degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.stack(T, dim = -1)
    
    def legendre_polynomials(self, x):
        T = [torch.ones_like(x), x]
        for n in range(2, self.chebyshev_degree + 1):
            T.append(((2 * (n - 1) + 1) * x * T[-1] - (n - 1) * T[-2]) / n)
        return torch.stack(T, dim = -1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        if self.use_layernorm:
            x = self.layernorm(x)

        if self.use_linear:
            if self.skip_activation:
                base_output = self.base_linear(x)
            else:
                base_output = self.base_linear(self.base_activation(x))
        else:
            base_output = 0

        if self.normalization == "min-max":
            x_mapped = 2 * (x - x.min(dim = 1, keepdim = True)[0]) / (x.max(dim = 1, keepdim = True)[0] - x.min(dim = 1, keepdim = True)[0] + 1e-8) - 1
        elif self.normalization == "tanh":
            x_mapped = torch.tanh(x)
        elif self.normalization == "standardization":
            x_mapped = (x - x.mean(dim = 1, keepdim = True)) / (x.std(dim = 1, keepdim = True) + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization}")
        
        if self.use_legendre:
            chebyshev_bases = self.legendre_polynomials(x_mapped)
        else:
            chebyshev_bases = self.chebyshev_polynomials(x_mapped)

        if self.enable_chebyshev_scaler:
            chebyshev_weight = self.chebyshev_weight * self.chebyshev_scaler.unsqueeze(-1)
        else:
            chebyshev_weight = self.chebyshev_weight
        chebyshev_output = torch.einsum('bic,oic->bo', chebyshev_bases, chebyshev_weight)

        return base_output + chebyshev_output