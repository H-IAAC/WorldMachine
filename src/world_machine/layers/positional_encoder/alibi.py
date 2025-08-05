import math
import warnings

import torch

from .positional_encoder import PositionalEncoder


def alibi_slopes(n_head):
    closest_power_of_2 = 2**math.floor(math.log2(n_head))

    bases = torch.tensor([n_head, 2*closest_power_of_2])

    ratio = (2**(-2**-(torch.log2(bases)-3)))
    slopes = ratio.unsqueeze(-1)*(ratio.unsqueeze(-1)
                                  ** torch.arange(0, bases.max(), 1))

    slopes = torch.concat(
        (slopes[0][:n_head], slopes[1][::2][:n_head-closest_power_of_2]))

    return slopes


class AlibiPositionalEncoder(PositionalEncoder):
    def __init__(self, embed_dim: int, sequence_size: int, n_head: int):
        super().__init__(embed_dim, sequence_size, n_head)

        self._create_m(n_head)

    def _create_m(self, n_head: int) -> torch.Tensor:
        if n_head < 1:
            self._m = None
        else:
            self.register_buffer("_m", alibi_slopes(n_head))
            self._m: torch.Tensor

    def _m_activation(self, m: torch.Tensor) -> torch.Tensor:
        return m

    def apply_attention_bias_pe(self, attention_bias: torch.Tensor) -> torch.Tensor:
        if self._m is None:
            warnings.warn(
                "AlibiPositionalEncoder instantiated without attention heads. Scores positional encoding will not be aplied.")
            return attention_bias

        device = attention_bias.device
        batch_size = attention_bias.shape[0]//self._n_head
        context_size = attention_bias.shape[-1]

        alibi_bias = torch.ones((context_size, context_size), device=device)
        alibi_bias = alibi_bias.tril()
        alibi_bias.fill_diagonal_(0)

        alibi_bias = alibi_bias.cumsum(0)

        alibi_bias = alibi_bias.unsqueeze(-1)

        alibi_bias = alibi_bias*self._m_activation(self._m)
        alibi_bias *= -1
        # alibi_bias: context, context, n_head

        alibi_bias = alibi_bias.repeat(1, 1, batch_size).permute(2, 0, 1)

        return attention_bias + alibi_bias


class LearnableAlibiPositionalEncoder(AlibiPositionalEncoder):

    def __init__(self, embed_dim, sequence_size, n_head):
        super().__init__(embed_dim, sequence_size, n_head)

        self._sigmoid = torch.nn.Sigmoid()

    def _m_activation(self, m: torch.Tensor) -> torch.Tensor:
        return self._sigmoid(m)

    def _create_m(self, n_head: int) -> torch.Tensor:
        self._m = torch.nn.Parameter(torch.Tensor(n_head))
