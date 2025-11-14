from world_machine import WorldMachine, WorldMachineBuilder
from world_machine.layers import PointwiseFeedforward


def get_benchmark_model() -> WorldMachine:
    builder = WorldMachineBuilder(128, 100, "alibi", False)

    builder.add_sensory_channel("channel0",
                                128,
                                PointwiseFeedforward(
                                    3, 2*128, output_dim=128),
                                PointwiseFeedforward(128, 2*128, output_dim=3))

    builder.add_sensory_channel("channel1",
                                128,
                                PointwiseFeedforward(
                                    3, 2*128, output_dim=128),
                                PointwiseFeedforward(128, 2*128, output_dim=3))

    builder.add_block(1, "channel0", n_attention_head=4)
    builder.add_block(1, "channel1", n_attention_head=1)

    builder.remove_positional_encoding = False
    builder.state_activation = "tanh"
    builder.state_dropout = False

    model = builder.build()

    return model
