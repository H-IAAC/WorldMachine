import abc

import torch

class ConditioningBlock(torch.nn.Module, abc.ABC):
    def __init__(self, embed_dim:int, conditioning_dim:int):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x:torch.Tensor, conditioning:torch.Tensor, 
                conditioning_mask:torch.Tensor|None=None) -> torch.Tensor:
        ...