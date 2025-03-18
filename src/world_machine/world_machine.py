import torch
from tensordict import TensorDict

from .layers import Clamp, SinePositionalEncoding


class WorldMachine(torch.nn.Module):
    def __init__(self, state_size: int, max_context_size: int,
                 blocks: torch.nn.ModuleList,
                 sensorial_encoders: torch.nn.ModuleDict | None = None,
                 sensorial_decoders: torch.nn.ModuleDict | None = None,
                 state_encoder: torch.nn.Module | None = None,
                 state_decoder: torch.nn.Module | None = None,
                 detach_decoders: set[str] = None,
                 use_positional_encoding: bool = True,
                 remove_positional_encoding: bool = False,
                 state_activation: str | None = "tanh"
                 ):
        super().__init__()

        self._max_context_size = max_context_size
        self._state_size = state_size

        self._blocks = blocks

        if sensorial_encoders is None:
            sensorial_encoders = torch.nn.ModuleDict()
        self._sensorial_encoders = sensorial_encoders

        if sensorial_decoders is None:
            sensorial_decoders = torch.nn.ModuleDict()
        self._sensorial_decoders = sensorial_decoders

        if state_encoder is None:
            state_encoder = torch.nn.Identity()
        self._state_encoder = state_encoder

        if state_decoder is None:
            state_decoder = torch.nn.Identity()
        self._state_decoder = state_decoder

        if detach_decoders is None:
            detach_decoders = set()
        self._detach_decoders = detach_decoders

        self._use_positional_encoding = use_positional_encoding
        self._remove_positional_encoding = remove_positional_encoding

        self._positional_encoder = SinePositionalEncoding(
            state_size, max_context_size)

        if state_activation is None:
            self._state_activation = torch.nn.Identity()
        elif state_activation == "tanh":
            self._state_activation = torch.nn.Tanh()
        elif state_activation == "clamp":
            self._state_activation = Clamp()
        else:
            raise ValueError(
                f"Invalid state activation function {state_activation}")

    def forward(self, state: torch.Tensor | None = None,
                state_decoded: torch.Tensor | None = None,
                sensorial_data: TensorDict | None = None,
                sensorial_masks: TensorDict | None = None,
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

        if sensorial_data is None:
            sensorial_data = TensorDict(device=device)

        if sensorial_masks is None:
            sensorial_masks = TensorDict(device=device, batch_size=batch_size)
            for name in sensorial_data.keys():
                sensorial_masks[name] = torch.ones(
                    (batch_size, seq_len), dtype=bool, device=device)

        x: TensorDict = sensorial_data.clone()

        for dim in x.keys():
            x[dim] = x[dim][:, :seq_len]

        # State encoding
        if state_decoded is not None:
            x["state"] = self._state_encoder(
                state_decoded[:, :seq_len])
        else:
            x["state"] = state[:, :seq_len].clone()

        if self._use_positional_encoding:
            x["state"] += self._positional_encoder()[:, :seq_len]

        # Sensorial encoding

        for name in self._sensorial_encoders:
            x[name] = self._sensorial_encoders[name](sensorial_data[name])

        y = x
        # Main prediction+update
        for block in self._blocks:
            y = block(y, sensorial_masks)

        if self._remove_positional_encoding and self._use_positional_encoding:
            y["state"] -= self._positional_encoder()[:, :seq_len]

        y["state"] = self._state_activation(y["state"])

        state: torch.Tensor = y["state"]
        state_detached = state.detach()

        # Sensorial decoding from state
        for name in self._sensorial_decoders:
            s = state
            if name in self._detach_decoders:
                s = state_detached

            y[name] = self._sensorial_decoders[name](s)

        # State decoding
        s = state
        if "state" in self._detach_decoders:
            s = state_detached
        y["state_decoded"] = self._state_decoder(s)

        return y

    def __call__(self, state: torch.Tensor | None = None,
                 state_decoded: torch.Tensor | None = None,
                 sensorial_data: TensorDict | None = None,
                 sensorial_masks: TensorDict | None = None,
                 input_sequence_size: int | None = None):
        return super().__call__(state, state_decoded, sensorial_data, sensorial_masks, input_sequence_size)
