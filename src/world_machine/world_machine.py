import torch
from tensordict import TensorDict

from .layers import SinePositionalEncoding


class WorldMachine(torch.nn.Module):
    def __init__(self, state_size: int, max_context_size: int,
                 blocks: torch.nn.ModuleList,
                 sensorial_encoders: torch.nn.ModuleDict | None = None,
                 sensorial_decoders: torch.nn.ModuleDict | None = None,
                 state_encoder: torch.nn.Module | None = None,
                 state_decoder: torch.nn.Module | None = None
                 ):
        super().__init__()

        self.blocks = blocks

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

        self._positional_encoder = SinePositionalEncoding(
            state_size, max_context_size)

    def forward(self, state: torch.Tensor,
                sensorial_data: TensorDict | None = None,
                sensorial_masks: TensorDict | None = None) -> TensorDict:

        device = state.device
        batch_size = state.shape[0]
        seq_len = state.shape[1]

        if sensorial_data is None:
            sensorial_data = TensorDict(device=device)

        if sensorial_masks is None:
            sensorial_masks = TensorDict(device=device, batch_size=batch_size)
            for name in sensorial_data.keys():
                sensorial_masks[name] = torch.ones(
                    (batch_size, seq_len), dtype=bool, device=device)

        # Sensorial encoding
        x: TensorDict = sensorial_data.copy()
        for name in self._sensorial_encoders:
            x[name] = self._sensorial_encoders[name](sensorial_data[name])

        # State encoding
        x["state"] = self._state_encoder(state) + self._positional_encoder()

        y = x
        # Main prediction+update
        for block in self.blocks:
            y = block(y, sensorial_masks)

        # ???
        # y["state"] -= self.positional_encoder()

        # Sensorial decoding from state
        for name in self._sensorial_decoders:
            y[name] = self._sensorial_decoders[name](y["state"])

        # State decoding
        y["state_decoded"] = self._state_decoder(y["state"])

        return y
