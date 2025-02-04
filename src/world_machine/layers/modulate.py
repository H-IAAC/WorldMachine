import torch

class Modulate(torch.nn.Module):
    def forward(self, x:torch.Tensor, scale:torch.Tensor, shift:torch.Tensor|None=None) -> torch.Tensor:
        if shift is None:
            return x * (1 + scale)
        else:
            return x * (1 + scale) + shift