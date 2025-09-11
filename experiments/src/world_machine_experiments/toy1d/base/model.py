import torch

from world_machine import WorldMachine, WorldMachineBuilder
from world_machine.layers import PointwiseFeedforward
from world_machine_experiments.toy1d.dimensions import Dimensions


def toy1d_model_untrained(block_configuration: list[Dimensions], state_dimensions: list[int] | None = None,
                          h_ensure_random_seed: None = None, remove_positional_encoding: bool = False, measurement_size: int = 2,
                          positional_encoder_type: str | None = "sine",
                          state_activation: str | None = None,
                          learn_sensorial_mask: bool = False,
                          state_size: int = 6,
                          n_attention_head: int = 1,
                          state_drouput: float | None = None) -> WorldMachine:

    decoded_state_size = len(
        state_dimensions) if state_dimensions is not None else 3

    max_context_size = 200

    builder = WorldMachineBuilder(state_size,
                                  max_context_size, positional_encoder_type,
                                  learn_sensorial_mask)

    if Dimensions.STATE_CONTROL in block_configuration:
        builder.add_sensorial_dimension("state_control", state_size,
                                        torch.nn.Linear(3, state_size),
                                        torch.nn.Linear(state_size, 3))

    if Dimensions.MEASUREMENT in block_configuration:
        builder.add_sensorial_dimension("measurement", state_size,
                                        PointwiseFeedforward(
                                            input_dim=measurement_size, hidden_size=2*state_size, output_dim=state_size),
                                        PointwiseFeedforward(
                                            input_dim=state_size, hidden_size=2*state_size, output_dim=measurement_size)
                                        )

    if Dimensions.STATE_DECODED in block_configuration:
        builder.add_sensorial_dimension("state_decoded", state_size,
                                        torch.nn.Linear(
                                            decoded_state_size, state_size),
                                        torch.nn.Linear(state_size, decoded_state_size))

    # torch.nn.Linear(decoded_state_size, state_size)
    builder.state_encoder = PointwiseFeedforward(
        input_dim=decoded_state_size, hidden_size=2*state_size, output_dim=state_size)
    # torch.nn.Linear(state_size, decoded_state_size)
    builder.state_decoder = PointwiseFeedforward(
        input_dim=state_size, hidden_size=2*state_size, output_dim=decoded_state_size)

    for config in block_configuration:
        if config == Dimensions.STATE:
            builder.add_block(n_attention_head=n_attention_head)
        elif config == Dimensions.STATE_CONTROL:
            builder.add_block(sensorial_dimension="state_control",
                              n_attention_head=n_attention_head)
        elif config == Dimensions.MEASUREMENT:
            builder.add_block(sensorial_dimension="measurement",
                              n_attention_head=n_attention_head)
        elif config == Dimensions.STATE_DECODED:
            builder.add_block(sensorial_dimension="state_decoded",
                              n_attention_head=n_attention_head)
        elif config == Dimensions.STATE_INPUT:
            builder.add_block(sensorial_dimension="state",
                              n_attention_head=n_attention_head)

    builder.remove_positional_encoding = remove_positional_encoding
    builder.state_activation = state_activation
    builder.state_dropout = state_drouput

    return builder.build()
