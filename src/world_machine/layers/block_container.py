import torch
from tensordict import TensorDict

from world_machine.current_version import CURRENT_COMPATIBILITY_VERSION
from world_machine.profile import profile_range
from world_machine.version_upgrader import upgrade


class BlockContainer(torch.nn.Module):
    def __init__(self, block: torch.nn.Module,
                 sensory_channel: str = ""):
        super().__init__()

        self._compatibility_version = CURRENT_COMPATIBILITY_VERSION

        self.block = block

        self.sensory_channel = sensory_channel

    @profile_range("block_container_forward", domain="world_machine")
    def forward(self, x: TensorDict,
                sensory_masks: TensorDict | None = None) -> TensorDict:

        state = x["state"]
        y = x.copy()

        if self.sensory_channel == "":
            y["state"] = self.block(state)
        else:
            sensory = x[self.sensory_channel]

            mask = None
            if sensory_masks is not None and self.sensory_channel in sensory_masks:
                mask = sensory_masks[self.sensory_channel]

            y["state"] = self.block(state, sensory, mask)

        return y

    def __setstate__(self, state):
        upgrade(state)
        return super().__setstate__(state)
