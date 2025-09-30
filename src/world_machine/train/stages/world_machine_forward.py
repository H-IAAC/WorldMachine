from tensordict import TensorDict

from world_machine import WorldMachine

from .train_stage import TrainStage


class WorldMachineForward(TrainStage):
    def __init__(self):
        super().__init__(0)

    def forward(self, model: WorldMachine, segment: TensorDict) -> None:
        sensorial_data = segment["inputs"]

        sensorial_masks = None
        if "input_masks" in segment:
            sensorial_masks = segment["input_masks"]

        if "state" in segment["inputs"]:
            state = segment["inputs"]["state"]

            logits: TensorDict = model(
                state=state, sensorial_data=sensorial_data, sensorial_masks=sensorial_masks)
        else:
            state_decoded = segment["inputs"]["state_decoded"]

            logits: TensorDict = model(
                state_decoded=state_decoded, sensorial_data=sensorial_data, sensorial_masks=sensorial_masks)

        segment["logits"] = logits
