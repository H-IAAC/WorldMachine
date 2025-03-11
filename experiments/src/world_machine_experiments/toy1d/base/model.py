import torch

from world_machine import WorldMachine, WorldMachineBuilder
from world_machine_experiments.toy1d.dimensions import Dimensions


def toy1d_model_untrained(block_configuration: list[Dimensions], state_dimensions: list[int] | None = None,
                          h_ensure_random_seed: None = None, remove_positional_encoding: bool = False, measurement_size: int = 2) -> WorldMachine:

    decoded_state_size = len(
        state_dimensions) if state_dimensions is not None else 3

    state_size = 6
    max_context_size = 200

    builder = WorldMachineBuilder(state_size,
                                  max_context_size)

    builder.add_sensorial_dimension("state_control", state_size,
                                    torch.nn.Linear(3, state_size),
                                    torch.nn.Linear(state_size, 3))

    builder.add_sensorial_dimension("next_measurement", state_size,
                                    torch.nn.Linear(
                                        measurement_size, state_size),
                                    torch.nn.Linear(state_size, measurement_size))

    builder.add_sensorial_dimension("state_decoded", state_size,
                                    torch.nn.Linear(3, state_size),
                                    torch.nn.Linear(state_size, 3))

    builder.state_encoder = torch.nn.Linear(decoded_state_size, state_size)
    builder.state_decoder = torch.nn.Linear(state_size, decoded_state_size)

    for config in block_configuration:
        if config == Dimensions.STATE:
            builder.add_block()
        elif config == Dimensions.STATE_CONTROL:
            builder.add_block(sensorial_dimension="state_control")
        elif config == Dimensions.NEXT_MEASUREMENT:
            builder.add_block(sensorial_dimension="next_measurement")
        elif config == Dimensions.STATE_DECODED:
            builder.add_block(sensorial_dimension="state_decoded")

    builder.remove_positional_encoding = remove_positional_encoding

    return builder.build()
