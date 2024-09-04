from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from thop import profile
from contextlib import redirect_stdout
import io
import math
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience = 20, min_delta = 0, monitor = "loss", path = "cifar100.pt", save_model = False):
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
        else:
            score = value

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

class DecreasingCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult = 1, eta_min = 0, last_epoch = -1, decay_factor = 0.5):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay_factor = decay_factor
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]
        self.restart_count = 0

    def get_lr(self):
        T_cur = self.T_cur
        if T_cur == 0 and self.last_epoch > 0:
            self.restart_count += 1

        decay = self.decay_factor ** self.restart_count
        return [self.eta_min + decay * (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / self.T_i)) / 2 for base_lr in self.initial_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()

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
            flops, params = profile(model, inputs = (input_tensor,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Parameters: {params / 1e6:.4f}M")