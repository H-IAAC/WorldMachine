import torch
from tensordict import TensorDict

from world_machine.layers.positional_encoder import create_positional_encoder

from .layers import Clamp, LTanh, Sine, SinTanh


def generate_sensorial_masks(sensorial_data: TensorDict) -> TensorDict:
    batch_size = sensorial_data.shape[0]
    seq_len = sensorial_data.shape[1]
    device = sensorial_data.device

    sensorial_masks = TensorDict(
        device=device, batch_size=[batch_size, seq_len])
    for name in sensorial_data.keys():
        sensorial_masks[name] = torch.ones(
            (batch_size, seq_len), dtype=bool, device=device)

    return sensorial_masks


class WorldMachine(torch.nn.Module):
    def __init__(self, state_size: int, max_context_size: int,
                 blocks: torch.nn.ModuleList,
                 sensorial_encoders: torch.nn.ModuleDict | None = None,
                 sensorial_decoders: torch.nn.ModuleDict | None = None,
                 state_encoder: torch.nn.Module | None = None,
                 state_decoder: torch.nn.Module | None = None,
                 detach_decoders: set[str] = None,
                 positional_encoder_type: str | None = "sine",
                 remove_positional_encoding: bool = False,
                 state_activation: str | None = "tanh",
                 state_dropout: float | None = None,
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
            sensorial_data = TensorDict(
                device=device, batch_size=[batch_size, seq_len])
        elif sensorial_data.shape[1] != seq_len:
            sensorial_data = sensorial_data[:, :seq_len]

        if sensorial_masks is None:
            sensorial_masks = generate_sensorial_masks(sensorial_data)
        if sensorial_masks.shape[1] != seq_len:
            sensorial_masks = sensorial_masks[:, :seq_len]

        x: TensorDict = sensorial_data.clone()

        # State encoding
        if state_decoded is not None:
            x["state"] = self._state_encoder(
                state_decoded[:, :seq_len])
        else:
            x["state"] = state[:, :seq_len].clone()

        x["state"] = self._positional_encoder.apply_input_pe(x["state"])

        if self._state_dropout is not None:
            x["state"] = self._state_dropout(x["state"])

        if input_sequence_size is not None:
            x = x.contiguous()

        # Sensorial encoding

        for name in self._sensorial_encoders:
            x[name] = self._sensorial_encoders[name](x[name])

        y = x
        # Main prediction+update
        for block in self._blocks:
            y = block(y, sensorial_masks)

        if self._remove_positional_encoding:
            y["state"] = self._positional_encoder.remove_input_pe(y["state"])

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
                 input_sequence_size: int | None = None) -> TensorDict:
        return super().__call__(state, state_decoded, sensorial_data, sensorial_masks, input_sequence_size)

    def inference(self,
                  state: torch.Tensor,
                  sensorial_data: TensorDict | None = None,
                  sensorial_masks: TensorDict | None = None,
                  start: int = 0,
                  total_size: int | None = None,
                  replace_sensorial_data: bool = False) -> TensorDict:
        '''
        Autoregressive inference.

        Args:
            state (torch.Tensor): The initial state tensor.
            sensorial_data (TensorDict | None, optional): Input sensorial data. Defaults to None.
            sensorial_masks (TensorDict | None, optional): Output sensorial data. Defaults to None.
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
            expansion = torch.empty([state.shape[0], total_size-state.shape[1],
                                    state.shape[2]], dtype=state.dtype, device=state.device)
            state = torch.hstack([state, expansion])

        expansion_seq_len = 0
        if sensorial_data is None:
            sensorial_data = TensorDict(device=device, batch_size=[
                                        batch_size, total_size])
        elif sensorial_data.batch_size[1] != total_size:
            expansion_seq_len = total_size-sensorial_data.batch_size[1]
            expansion = TensorDict(device=device, batch_size=[
                                   batch_size, expansion_seq_len])

            for key in sensorial_data.keys():
                expansion[key] = torch.empty(
                    [batch_size, expansion_seq_len]+list(sensorial_data[key].shape[2:]), device=device)

            sensorial_data = torch.cat([sensorial_data, expansion], dim=1)

        if sensorial_masks is None:
            sensorial_masks = generate_sensorial_masks(sensorial_data)

            if expansion_seq_len != 0:
                for key in sensorial_data.keys():
                    sensorial_masks[key][:, -expansion_seq_len:] = 0

        if replace_sensorial_data:
            sensorial_data = sensorial_data.clone()

        state = state.clone().to(device)
        sensorial_data = sensorial_data.to(device)
        sensorial_masks = sensorial_masks.to(device)

        for i in range(start, total_size):
            logits = self(state=state, sensorial_data=sensorial_data,
                          sensorial_masks=sensorial_masks, input_sequence_size=i+1)

            if i != total_size-1:
                state[:, i+1] = logits["state"][:, i]

                if replace_sensorial_data:
                    for name in sensorial_data.keys():
                        sensorial_data[name][:, i+1] = logits[name][:, i]

        return logits
