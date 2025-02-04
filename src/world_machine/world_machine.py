import torch
from tensordict import TensorDict

from .layers import SinePositionalEncoding

class WorldMachine(torch.nn.Module):
    def __init__(self, state_dim, max_context_size:int):
        super().__init__()

        self.sensorial_encoders : torch.nn.ModuleDict
        self.blocks : torch.nn.ModuleList
        self.sensorial_decoders : torch.nn.ModuleDict

        self.state_encoder : torch.nn.Module = torch.nn.Identity()
        self.state_decoder : torch.nn.Module = torch.nn.Identity()

        self.positional_encoder = SinePositionalEncoding(state_dim, max_context_size)

    def forward(self, sensorial_data:TensorDict, 
                sensorial_masks:TensorDict,
                state:torch.Tensor) -> TensorDict:

        #Sensorial encoding
        x = sensorial_data.copy()        
        for name in self.sensorial_encoders:
            x[name] = self.sensorial_encoders[name](sensorial_data[name])

        
        #State encoding
        x["state"] = self.state_encoder(state) + self.positional_encoder()

        y = x
        #Main prediction+update
        for block in self.blocks:
            y = block(y, sensorial_masks)

        #???
        #y["state"] -= self.positional_encoder()

        #Sensorial decoding from state
        for name in self.sensorial_decoders:
            y[name] = self.sensorial_decoders[name](y["state"])
        
        #State decoding
        y["state_decoded"] = self.state_decoder(y["state"])

        return y