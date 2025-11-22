import torch

from world_machine.current_version import CURRENT_COMPATIBILITY_VERSION
from world_machine.profile import profile_range
from world_machine.version_upgrader import upgrade

from .attention import MultiHeadSelfAttention
from .conditioning_block import ConditioningBlock
from .modulate import Modulate


class AdaLNZeroBlock(ConditioningBlock):
    def __init__(self, embed_dim: int, conditioning_dim: int, hidden_size: int,
                 n_head: int, dropout_rate: float = 0.0, positional_encoder_type: str | None = None, learn_sensory_mask: bool = False):
        super().__init__(embed_dim, conditioning_dim)

        self._compatibility_version = CURRENT_COMPATIBILITY_VERSION

        self.conditioning_mlp = torch.nn.Sequential(torch.nn.SiLU(),
                                                    torch.nn.Linear(conditioning_dim, 6*embed_dim, bias=True))

        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.modulate1 = Modulate()
        self.attention = MultiHeadSelfAttention(
            embed_dim, n_head, True, positional_encoder_type)
        self.dropout_attention = torch.nn.Dropout(dropout_rate)
        self.modulate2 = Modulate()

        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        self.modulate3 = Modulate()
        self.linear1 = torch.nn.Linear(embed_dim, hidden_size)
        self.dropout_linear1 = torch.nn.Dropout(dropout_rate)
        self.act = torch.nn.GELU(approximate="tanh")
        self.linear2 = torch.nn.Linear(hidden_size, embed_dim)
        self.dropout_linear2 = torch.nn.Dropout(dropout_rate)
        self.modulate4 = Modulate()

        if learn_sensory_mask:
            self.sensory_replace = torch.nn.Parameter(torch.Tensor(36))
        else:
            self.sensory_replace = None

    @profile_range("adaln_zero_block_forward", domain="world_machine")
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor,
                conditioning_mask: torch.Tensor | None = None) -> torch.Tensor:

        context_size = x.shape[1]

        # Conditioning MLP
        with profile_range("conditioning_mlp", category="adaln_zero", domain="world_machine"):
            conditioning_data = self.conditioning_mlp(conditioning)[
                :, :context_size]

            if conditioning_mask is not None:
                if self.sensory_replace is None:
                    conditioning_data *= conditioning_mask.unsqueeze(-1)
                else:
                    conditioning_data[torch.bitwise_not(
                        conditioning_mask)] = self.sensory_replace

            # scale -> gamma, alpha
            # shift -> beta
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = conditioning_data.chunk(
                6, dim=2)

        # gamma1: torch.Tensor
        # beta1: torch.Tensor
        # alpha1: torch.Tensor
        # gamma2: torch.Tensor
        # beta2: torch.Tensor
        # alpha2: torch.Tensor

        # First part (before first +)
        y1 = self.layer_norm1(x)
        y1 = self.modulate1(y1, shift=gamma1, scale=beta1)
        y1 = self.dropout_attention(self.attention(y1))
        y1 = self.modulate2(y1, scale=alpha1)

        # First +
        y1 = y1+x

        # Second part
        y2 = self.layer_norm2(y1)
        y2 = self.modulate3(y2, scale=gamma2, shift=beta2)

        y2 = self.linear1(y2)
        y2 = self.dropout_linear1(self.act(y2))
        y2 = self.dropout_linear2(self.linear2(y2))

        y2 = self.modulate4(y2, scale=alpha2)

        # Second +
        y2 = y2+y1

        return y2

    def __setstate__(self, state):
        upgrade(state)
        return super().__setstate__(state)
