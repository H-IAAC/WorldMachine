import torch


class Sine(torch.nn.Module):
    def __init__(self, min: float = -1, max: float = 1):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
