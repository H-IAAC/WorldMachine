import torch


class LTanh(torch.nn.Module):
    def __init__(self, size):
        super().__init__()

        self._alpha = torch.nn.Parameter(torch.Tensor(size))
        self._tanh = torch.nn.Tanh()

        torch.nn.init.kaiming_normal_(self._alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._alpha*x
        y = self._tanh(y)

        return y
