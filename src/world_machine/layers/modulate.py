import torch

from world_machine.profile import profile_range


class Modulate(torch.nn.Module):
    @profile_range("modulate_forward", domain="world_machine")
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor | None = None) -> torch.Tensor:
        if shift is None:
            return x * (1 + scale)
        else:
            return x * (1 + scale) + shift
