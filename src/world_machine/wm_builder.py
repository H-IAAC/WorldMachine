import torch

from .layers import AdaLNZeroBlock, BlockContainer, TransformDecoderBlock
from .world_machine import WorldMachine


class WorldMachineBuilder:
    def __init__(self, state_size: int, max_context_size: int, positional_encoder_type: str | None = "alibi",
                 learn_sensory_mask: bool = False):
        self._state_size = state_size
        self._max_context_size = max_context_size
        self._positional_encoder_type = positional_encoder_type

        self._sensory_channels: dict[str, int] = {}
        self._sensory_encoders: dict[str, torch.nn.Module] = {}
        self._sensory_decoders: dict[str, torch.nn.Module] = {}

        self._blocks: list[BlockContainer] = []

        self._state_encoder = torch.nn.Identity()
        self._state_decoder = torch.nn.Identity()

        self._detach_decoder: set[str] = set()

        self.remove_positional_encoding = False
        self._state_activation: str | None = "tanh"

        self._learn_sensory_mask = learn_sensory_mask

        self._state_dropout: float | None = None

    @property
    def state_encoder(self) -> torch.nn.Module:
        return self._state_encoder

    @state_encoder.setter
    def state_encoder(self, encoder: torch.nn.Module):
        self._state_encoder = encoder

    @property
    def state_decoder(self) -> torch.nn.Module:
        return self._state_decoder

    @state_decoder.setter
    def state_decoder(self, decoder: torch.nn.Module):
        self._state_decoder = decoder

    @property
    def detach_state_decoder(self) -> bool:
        return "state" in self._detach_decoder

    @detach_state_decoder.setter
    def detach_state_decoder(self, value: bool) -> None:
        self._detach_decoder.add("state")

    @property
    def state_activation(self) -> str | None:
        return self._state_activation

    @state_activation.setter
    def state_activation(self, value) -> None:
        if value not in [None, "tanh", "clamp", "ltanh", "sintanh", "sin"]:
            raise ValueError(f"Invalid state activation function {value}.")

        self._state_activation = value

    @property
    def state_dropout(self) -> float | None:
        return self._state_dropout

    @state_dropout.setter
    def state_dropout(self, value: float | None) -> None:
        if value is None or isinstance(value, float):
            self._state_dropout = value

    @property
    def learn_sensory_mask(self) -> bool:
        return self._learn_sensory_mask

    def add_sensory_channel(self, channel_name: str, channel_size: int,
                            encoder: torch.nn.Module | None = None,
                            decoder: torch.nn.Module | None = None, detach_decoder: bool = False):

        assert channel_name != "state"

        self._sensory_channels[channel_name] = channel_size

        if encoder is not None:
            self._sensory_encoders[channel_name] = encoder
        if decoder is not None:
            self._sensory_decoders[channel_name] = decoder

        if detach_decoder:
            self._detach_decoder.add(channel_name)

    def add_block(self, count: int = 1, sensory_channel: str = "",
                  dropout_rate: float = 0.1, hidden_size_multiplier: int = 4,
                  n_attention_head: int = 1):
        for _ in range(count):
            if sensory_channel == "":
                block = TransformDecoderBlock(self._state_size,
                                              hidden_size_multiplier*self._state_size,
                                              n_attention_head,
                                              dropout_rate,
                                              is_causal=True,
                                              positional_encoder_type=self._positional_encoder_type)
            else:
                if sensory_channel == "state":
                    channel_size = self._state_size
                else:
                    channel_size = self._sensory_channels[sensory_channel]

                block = AdaLNZeroBlock(self._state_size,
                                       channel_size,
                                       hidden_size_multiplier*self._state_size,
                                       n_attention_head,
                                       positional_encoder_type=self._positional_encoder_type,
                                       learn_sensory_mask=self._learn_sensory_mask)

            self._blocks.append(BlockContainer(
                block, sensory_channel=sensory_channel))

    def build(self) -> WorldMachine:
        wm = WorldMachine(self._state_size,
                          self._max_context_size,
                          torch.nn.ModuleList(self._blocks),
                          torch.nn.ModuleDict(self._sensory_encoders),
                          torch.nn.ModuleDict(self._sensory_decoders),
                          self._state_encoder,
                          self._state_decoder,
                          self._detach_decoder,
                          self._positional_encoder_type,
                          self.remove_positional_encoding,
                          self._state_activation,
                          self._state_dropout)

        return wm
