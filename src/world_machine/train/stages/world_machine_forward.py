from tensordict import TensorDict

from world_machine.train.mode import DatasetPassMode
from world_machine.world_machine import WorldMachine

from .train_stage import TrainStage


class WorldMachineForward(TrainStage):
    def __init__(self):
        super().__init__(0)

    def forward(self, model: WorldMachine, segment: TensorDict, mode: DatasetPassMode) -> None:
        sensory_data = segment["inputs"]

        sensory_masks = None
        if "input_masks" in segment:
            sensory_masks = segment["input_masks"]

        if "state" in segment["inputs"]:
            state = segment["inputs"]["state"]

            logits: TensorDict = model(
                state=state, sensory_data=sensory_data, sensory_masks=sensory_masks)
        else:
            state_decoded = segment["inputs"]["state_decoded"]

            logits: TensorDict = model(
                state_decoded=state_decoded, sensory_data=sensory_data, sensory_masks=sensory_masks)

        segment["logits"] = logits
