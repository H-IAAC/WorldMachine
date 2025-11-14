from typing import Any

import torch
from tensordict import TensorDict

from world_machine.layers import BlockContainer
from world_machine.layers.positional_encoder import create_positional_encoder
from world_machine.profile import profile_range
from world_machine.version_upgrader import upgrade

from .current_version import CURRENT_COMPATIBILITY_VERSION
from .layers import Clamp, LTanh, MultiHeadAttention, Sine, SinTanh


@profile_range("generate_sensory_masks", domain="world_machine")
def generate_sensory_masks(sensory_data: TensorDict) -> TensorDict:
    batch_size = sensory_data.shape[0]
    seq_len = sensory_data.shape[1]
    device = sensory_data.device

    sensory_masks = TensorDict(
        device=device, batch_size=[batch_size, seq_len])
    for name in sensory_data.keys():
        sensory_masks[name] = torch.ones(
            (batch_size, seq_len), dtype=bool, device=device)

    return sensory_masks


class WorldMachine(torch.nn.Module):
    def __init__(self, state_size: int, max_context_size: int,
                 blocks: torch.nn.ModuleList,
                 sensory_encoders: torch.nn.ModuleDict | None = None,
                 sensory_decoders: torch.nn.ModuleDict | None = None,
                 state_encoder: torch.nn.Module | None = None,
                 state_decoder: torch.nn.Module | None = None,
                 detach_decoders: set[str] = None,
                 positional_encoder_type: str | None = "sine",
                 remove_positional_encoding: bool = False,
                 state_activation: str | None = "tanh",
                 state_dropout: float | None = None,
                 ):
        super().__init__()
        self._compatibility_version = CURRENT_COMPATIBILITY_VERSION

        self._max_context_size = max_context_size
        self._state_size = state_size

        self._blocks = blocks

        sensory_channels: set[str] = set()
        for block in blocks:
            block: BlockContainer
            sensory_channels.add(block.sensory_channel)

            if block.sensory_channel == "state":
                block.sensory_channel = "state_input"

        self._sensory_channels = sensory_channels

        if sensory_encoders is None:
            sensory_encoders = torch.nn.ModuleDict()
        self._sensory_encoders = sensory_encoders

        if sensory_decoders is None:
            sensory_decoders = torch.nn.ModuleDict()
        self._sensory_decoders = sensory_decoders

        if state_encoder is None:
            state_encoder = torch.nn.Identity()
        self._state_encoder = state_encoder

        if state_decoder is None:
            state_decoder = torch.nn.Identity()
        self._state_decoder = state_decoder

        if detach_decoders is None:
            detach_decoders = set()
        self._detach_decoders = detach_decoders

        self._positional_encoder = create_positional_encoder(
            positional_encoder_type, state_size, max_context_size)

        self._positional_encoder_type = positional_encoder_type
        self._remove_positional_encoding = remove_positional_encoding

        if state_activation is None:
            self._state_activation = torch.nn.Identity()
        elif state_activation == "tanh":
            self._state_activation = torch.nn.Tanh()
        elif state_activation == "clamp":
            self._state_activation = Clamp()
        elif state_activation == "ltanh":
            self._state_activation = LTanh(state_size)
        elif state_activation == "sintanh":
            self._state_activation = SinTanh()
        elif state_activation == "sin":
            self._state_activation = Sine()
        else:
            raise ValueError(
                f"Invalid state activation function {state_activation}")

        if state_dropout is not None:
            state_dropout = torch.nn.Dropout(state_dropout)
        self._state_dropout = state_dropout

        self._local_mode = False

    @property
    def local_mode(self) -> bool:
        return self._local_mode

    @local_mode.setter
    def local_mode(self, value: bool) -> None:
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                module.local_only = value

        self._local_mode = value

    @profile_range("pre_compute_attention_bias", category="world_machine", domain="world_machine")
    def pre_compute_attention_bias(self, size: int) -> None:
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                module.pre_compute_attention_bias(size)

    @profile_range("world_machine_forward", domain="world_machine")
    def forward(self, state: torch.Tensor | None = None,
                state_decoded: torch.Tensor | None = None,
                sensory_data: TensorDict | None = None,
                sensory_masks: TensorDict | None = None,
                input_sequence_size: int | None = None) -> TensorDict:

        if state_decoded is not None:
            device = state_decoded.device
            batch_size = state_decoded.shape[0]
            seq_len = state_decoded.shape[1]
        elif state is not None:
            device = state.device
            batch_size = state.shape[0]
            seq_len = state.shape[1]
        else:
            raise ValueError(
                "'state_decoded' or 'state' must but not None, but both is None.")

        if input_sequence_size is not None:
            seq_len = input_sequence_size

        with profile_range("prepare_sensory_data",
                           category="main_forward", domain="world_machine"):
            if sensory_data is None:
                sensory_data = TensorDict(
                    device=device, batch_size=[batch_size, seq_len])
            elif sensory_data.shape[1] != seq_len:
                sensory_data = sensory_data[:, :seq_len]

        with profile_range("prepare_sensory_masks",
                           category="main_forward", domain="world_machine"):
            if sensory_masks is not None and sensory_masks.shape[1] != seq_len:
                sensory_masks = sensory_masks[:, :seq_len]

        with profile_range("clone_sensory_data",
                           category="main_forward", domain="world_machine"):
            x: TensorDict = sensory_data.clone()

        # State encoding
        with profile_range("state_encoding",
                           category="main_forward", domain="world_machine"):
            if state_decoded is not None:
                x["state"] = self._state_encoder(
                    state_decoded[:, :seq_len])
            else:
                x["state"] = state[:, :seq_len].clone()

        with profile_range("add_positional_encoder",
                           category="main_forward", domain="world_machine"):
            x["state"] = self._positional_encoder.apply_input_pe(x["state"])

        if self._state_dropout is not None:
            x["state"] = self._state_dropout(x["state"])

        if "state" in self._sensory_channels:
            state_input: torch.Tensor = x["state"].clone()
            x["state_input"] = state_input

        if input_sequence_size is not None:
            x = x.contiguous()

        # Sensory encoding
        with profile_range("sensory_encoding",
                           category="main_forward", domain="world_machine"):
            for name in self._sensory_encoders:
                x[name] = self._sensory_encoders[name](x[name])

        y = x
        # Main prediction+update
        with profile_range("blocks",
                           category="main_forward", domain="world_machine"):
            for block in self._blocks:
                y = block(y, sensory_masks)

        with profile_range("remove_positional_encoder",
                           category="main_forward", domain="world_machine"):
            if self._remove_positional_encoding:
                y["state"] = self._positional_encoder.remove_input_pe(
                    y["state"])

        with profile_range("state_activation",
                           category="main_forward", domain="world_machine"):
            y["state"] = self._state_activation(y["state"])

        state: torch.Tensor = y["state"]
        state_detached = state.detach()

        # Sensory decoding from state
        with profile_range("sensory_decoding",
                           category="main_forward", domain="world_machine"):
            for name in self._sensory_decoders:
                s = state
                if name in self._detach_decoders:
                    s = state_detached

                y[name] = self._sensory_decoders[name](s)

        # State decoding
        with profile_range("state_decoding",
                           category="main_forward", domain="world_machine"):

            s = state
            if "state" in self._detach_decoders:
                s = state_detached
            y["state_decoded"] = self._state_decoder(s)

        return y

    def __call__(self, state: torch.Tensor | None = None,
                 state_decoded: torch.Tensor | None = None,
                 sensory_data: TensorDict | None = None,
                 sensory_masks: TensorDict | None = None,
                 input_sequence_size: int | None = None) -> TensorDict:
        return super().__call__(state, state_decoded, sensory_data, sensory_masks, input_sequence_size)

    @profile_range("world_machine_inference", domain="world_machine")
    def inference(self,
                  state: torch.Tensor,
                  sensory_data: TensorDict | None = None,
                  sensory_masks: TensorDict | None = None,
                  start: int = 0,
                  total_size: int | None = None,
                  replace_sensory_data: bool = False) -> TensorDict:
        '''
        Autoregressive inference.

        Args:
            state (torch.Tensor): The initial state tensor.
            sensory_data (TensorDict | None, optional): Input sensory data. Defaults to None.
            sensory_masks (TensorDict | None, optional): Output sensory data. Defaults to None.
            start (int, optional): Index to start the inference. Defaults to 0.
            total_size (int | None, optional): Total size of the sequence, if None, uses the state sequence length. Defaults to None.

        Returns:
            TensorDict: _description_
        '''

        device = next(iter(self.parameters())).device
        batch_size = state.shape[0]

        if total_size is None:
            total_size = state.shape[1]

        elif state.shape != total_size:
            with profile_range("expand_state", category="main_inference", domain="world_machine"):
                expansion = torch.empty([state.shape[0], total_size-state.shape[1],
                                        state.shape[2]], dtype=state.dtype, device=state.device)
                state = torch.hstack([state, expansion])

        with profile_range("prepare_sensory_data", category="main_inference", domain="world_machine"):
            expansion_seq_len = 0
            if sensory_data is None:
                sensory_data = TensorDict(device=device, batch_size=[
                    batch_size, total_size])
            elif sensory_data.batch_size[1] != total_size:
                expansion_seq_len = total_size-sensory_data.batch_size[1]
                expansion = TensorDict(device=device, batch_size=[
                    batch_size, expansion_seq_len])

            if expansion_seq_len != 0:
                with profile_range("expand_sensory_data", category="main_inference", domain="world_machine"):
                    for key in sensory_data.keys():
                        expansion[key] = torch.empty(
                            [batch_size, expansion_seq_len]+list(sensory_data[key].shape[2:]), device=device)

                    sensory_data = torch.cat(
                        [sensory_data, expansion], dim=1)

        if sensory_masks is None:
            sensory_masks = generate_sensory_masks(sensory_data)

            if expansion_seq_len != 0:
                with profile_range("expand_sensory_mask", category="main_inference", domain="world_machine"):
                    for key in sensory_data.keys():
                        sensory_masks[key][:, -expansion_seq_len:] = 0

        with profile_range("clone_transfer_data", category="main_inference", domain="world_machine"):
            if replace_sensory_data:
                sensory_data = sensory_data.clone()

            state = state.clone().to(device)
            sensory_data = sensory_data.to(device)
            sensory_masks = sensory_masks.to(device)

        for i in range(start, total_size):
            with profile_range("inference_step", category="main_inference", domain="world_machine"):
                logits = self(state=state, sensory_data=sensory_data,
                              sensory_masks=sensory_masks, input_sequence_size=i+1)

                if i != total_size-1:
                    state[:, i+1] = logits["state"][:, i]

                    if replace_sensory_data:
                        for name in sensory_data.keys():
                            sensory_data[name][:, i+1] = logits[name][:, i]

        return logits

    def __setstate__(self, state: dict[str, Any]) -> None:
        upgrade(state)
        super().__setstate__(state)
