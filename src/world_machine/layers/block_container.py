import torch
from tensordict import TensorDict

from world_machine.profile import profile_range


class BlockContainer(torch.nn.Module):
    def __init__(self, block: torch.nn.Module,
                 sensorial_dimension: str = ""):
        super().__init__()

        self.block = block

        self.sensorial_dimension = sensorial_dimension

    @profile_range("block_container_forward", domain="world_machine")
    def forward(self, x: TensorDict,
                sensorial_masks: TensorDict) -> TensorDict:

        state = x["state"]
        y = x.copy()

        if self.sensorial_dimension == "":
            y["state"] = self.block(state)
        else:
            sensorial = x[self.sensorial_dimension]
            mask = sensorial_masks[self.sensorial_dimension]
            y["state"] = self.block(state, sensorial, mask)

        return y
