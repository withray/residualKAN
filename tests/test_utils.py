from thop import profile
from contextlib import redirect_stdout
import io
import torch
import torch.nn as nn
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 16, pool = "avg", gate = "sigmoid"):
        super(SEBlock, self).__init__()
        self.pool = pool
        self.gate = gate
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_avg = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias = False),
        )
        self.fc_max = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias = False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias = False),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softsign = nn.Softsign()

    def forward(self, x):
        if self.pool == "avg":
            pooled = self.avg_pool(x)
            out = self.fc_avg(pooled)
        elif self.pool == "max":
            pooled = self.max_pool(x)
            out = self.fc_max(pooled)
        elif self.pool == "both":
            avg_out = self.fc_avg(self.avg_pool(x))
            max_out = self.fc_max(self.max_pool(x))
            out = avg_out + max_out
        else:
            raise ValueError(f"Invalid pool type: {self.pool}. Choose from 'avg', 'max', or 'both'.")
        
        if self.gate == "sigmoid":
            out = self.sigmoid(out)
        elif self.gate == "tanh":
            out = self.tanh(out)
        elif self.gate == "softsign":
            out = self.softsign(out)
        else:
            raise ValueError(f"Invalid gate type: {self.gate}. Choose from 'sigmoid', 'tanh', or 'softsign'.")
        return x * out

class EarlyStopping:
    def __init__(self, patience = 100, min_delta = 0, monitor = "loss", path = "best.pt", save_model = True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.monitor = monitor
        self.save_model = save_model

    def __call__(self, value, model):
        if self.monitor == "loss":
            score = -value
        elif self.monitor == "accuracy":
            score = value
        else:
            raise ValueError(f"Invalid monitor value '{self.monitor}'.")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.save_model:
            torch.save(model, self.path)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size

def warmup_scheduler(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def profile_model(model, input_size, device):
    model.eval()
    input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
    captured_output = io.StringIO()
    with redirect_stdout(captured_output):
        with torch.no_grad():
            flops, params, *_ = profile(model, inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Parameters: {params / 1e6:.4f}M")

def cutmix_data(x, y, alpha = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))

    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha = 0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam