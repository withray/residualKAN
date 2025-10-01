import torch
import math

class ChebyshevKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        polynomial_degree = 3,
        enable_scaler = True,
        base_activation = torch.nn.SiLU,
        use_linear = True,
        skip_activation = True,
        normalization = "tanh",
        polynomial_type = "chebyshev",
        use_layernorm = False,
        jacobi_alpha = 0.5,
        jacobi_beta = 0.5,
    ):
        super(ChebyshevKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.polynomial_degree = polynomial_degree
        self.use_linear = use_linear
        self.skip_activation = skip_activation
        self.normalization = normalization
        self.polynomial_type = polynomial_type
        self.use_layernorm = use_layernorm
        self.jacobi_alpha = jacobi_alpha
        self.jacobi_beta = jacobi_beta

        if self.use_linear:
            self.base_linear = torch.nn.Linear(in_features, out_features, bias = False)
        self.polynomial_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, polynomial_degree + 1))
        if enable_scaler:
            self.scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if self.normalization == "arctan":
            self.arctan_scale = torch.nn.Parameter(torch.tensor(0.1)) 

        self.enable_scaler = enable_scaler
        self.base_activation = base_activation()

        self.reset_parameters()
        if self.use_layernorm:
            self.layernorm = torch.nn.LayerNorm(out_features)

    def reset_parameters(self):
        with torch.no_grad():
            std_dev = 1.0 / math.sqrt(self.in_features * (self.polynomial_degree + 1))
            torch.nn.init.normal_(self.polynomial_weight, mean = 0.0, std = std_dev)
            if self.use_linear:
                torch.nn.init.kaiming_normal_(self.base_linear.weight, nonlinearity = "relu")
            if self.enable_scaler:
                torch.nn.init.constant_(self.scaler, 1.0)

    def chebyshev_polynomials(self, x):
        T = [torch.ones_like(x), x]
        for _ in range(2, self.polynomial_degree + 1):
            T.append(2 * x * T[-1] - T[-2])
        return torch.stack(T, dim = -1)
    
    def legendre_polynomials(self, x):
        T = [torch.ones_like(x), x]
        for n in range(2, self.polynomial_degree + 1):
            T.append(((2 * (n - 1) + 1) * x * T[-1] - (n - 1) * T[-2]) / n)
        return torch.stack(T, dim = -1)
    
    def jacobi_polynomials(self, x, alpha = 0.5, beta = 0.5):
        T = [torch.ones_like(x), 0.5 * (alpha - beta + (alpha + beta + 2) * x)]
        for n in range(2, self.polynomial_degree + 1):
            A = (2 * n + alpha + beta - 1) * ((2 * n + alpha + beta) * (2 * n + alpha + beta - 2) * x + alpha ** 2 - beta ** 2)
            B = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + alpha + beta)
            C = 2 * n * (n + alpha + beta) * (2 * n + alpha + beta - 2)
            T.append((A * T[-1] - B * T[-2]) / C)
        return torch.stack(T, dim = -1)
    
    def hermite_polynomials(self, x):
        T = [torch.ones_like(x), 2 * x]
        for n in range(2, self.polynomial_degree + 1):
            T.append(2 * x * T[-1] - 2 * (n - 1) * T[-2])
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
        elif self.normalization == "arctan":
            x_mapped = (2 / math.pi) * torch.atan((x - x.mean(dim = 1, keepdim = True)) * self.arctan_scale)
        elif self.normalization == "standardization":
            x_mapped = (x - x.mean(dim = 1, keepdim = True)) / (x.std(dim = 1, keepdim = True) + 1e-8)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization}")
        
        if self.polynomial_type == "legendre":
            polynomial_bases = self.legendre_polynomials(x_mapped)
        elif self.polynomial_type == "chebyshev":
            polynomial_bases = self.chebyshev_polynomials(x_mapped)
        elif self.polynomial_type == "hermite":
            polynomial_bases = self.hermite_polynomials(x_mapped)
        elif self.polynomial_type == "jacobi":
            polynomial_bases = self.jacobi_polynomials(x_mapped, alpha = self.jacobi_alpha, beta = self.jacobi_beta)
        else:
            raise ValueError(f"Unsupported polynomial type: {self.polynomial_type}")

        if self.enable_scaler:
            polynomial_weight = self.polynomial_weight * self.scaler.unsqueeze(-1)
        else:
            polynomial_weight = self.polynomial_weight
        polynomial_output = torch.einsum("bic,oic->bo", polynomial_bases, polynomial_weight)

        return base_output + polynomial_output