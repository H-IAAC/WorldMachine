import torch


class Clamp(torch.nn.Module):
    def __init__(self, min: float = -1, max: float = 1):
        super().__init__()

        self._min = min
        self._max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self._min, self._max)
