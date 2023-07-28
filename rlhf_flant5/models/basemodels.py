import torch

class VarianceLayerNorm(torch.nn.Module):
    def __init__(self, layer_size: int):
        self.layer_size = layer_size
        self.eps = 1e-6

    def forward(self, inputs: torch.Tensor):
        variance = inputs.pow(2).sum(-1)