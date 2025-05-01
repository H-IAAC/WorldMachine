import torch


class SinTanh(torch.nn.Module):
    def __init__(self, min: float = -1, max: float = 1):
        super().__init__()

        self._tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi*self._tanh(x))
