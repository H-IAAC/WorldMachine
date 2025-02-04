import torch

from .layers import BlockContainer, TransformDecoderBlock, AdaLNZeroBlock
from .world_machine import WorldMachine

class WorldMachineBuilder:
    def __init__(self, state_size: int, max_context_size: int):
        self._state_size = state_size
        self._max_context_size = max_context_size

        self._sensorial_dimensions: dict[str, int] = {}
        self._sensorial_encoders : dict[str, torch.nn.Module] = {}
        self._sensorial_decoders : dict[str, torch.nn.Module] = {}

        self._blocks: list[BlockContainer] = []

        self._state_encoder = torch.nn.Identity()

    @property
    def state_encoder(self) -> torch.nn.Module:
        return self._state_encoder
    
    @state_encoder.setter
    def state_encoder(self, encoder:torch.nn.Module):
        self._state_encoder = encoder

    @property
    def state_decoder(self) -> torch.nn.Module:
        return self._state_decoder
    
    @state_decoder.setter
    def state_decoder(self, decoder:torch.nn.Module):
        self._state_decoder = decoder


    def add_sensorial_dimension(self, dimension_name: str, dimension_size: int,
                                encoder:torch.nn.Module|None=None, decoder:torch.nn.Module|None=None):
        self._sensorial_dimensions[dimension_name] = dimension_size

        if encoder is not None:
            self._sensorial_encoders[dimension_name] = encoder
        if decoder is not None:
            self._sensorial_decoders[dimension_name] = decoder

    def add_block(self, count: int = 1, sensorial_dimension: str = "",
                  dropout_rate: float = 0.1, hidden_size_multiplier: int = 4,
                  n_attention_head:int=1):
        for _ in range(count):
            if sensorial_dimension == "":
                block = TransformDecoderBlock(self._state_size,
                                              hidden_size_multiplier*self._state_size,
                                              n_attention_head,
                                              dropout_rate,
                                              is_causal=True)
            else:
                block = AdaLNZeroBlock(self._state_size,
                                       self._sensorial_dimensions[sensorial_dimension],
                                       hidden_size_multiplier*self._state_size,
                                       n_attention_head)

            self._blocks.append(BlockContainer(
                block, sensorial_dimension=sensorial_dimension))

    def create_model(self) -> WorldMachine:
        wm = WorldMachine(self._state_size, self._max_context_size)
        
        wm.sensorial_encoders = torch.nn.ModuleDict(self._sensorial_encoders)
        wm.sensorial_decoders = torch.nn.ModuleDict(self._sensorial_decoders)

        wm.blocks = torch.nn.Sequential(*self._blocks)

        wm.state_encoder = self._state_encoder
        wm.state_decoder = self._state_decoder