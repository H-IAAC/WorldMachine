import torch

from world_machine.profile import profile_range

from .attention import MultiHeadSelfAttention
from .pointwise_feedforward import PointwiseFeedforward


class TransformDecoderBlock(torch.nn.Module):
    """
    Block of a Transform Decoder.
    """

    def __init__(self, embed_dim: int, hidden_size: int, n_head: int,
                 dropout_rate: float = 0.0, is_causal: bool = True, positional_encoder_type: str | None = None):
        super().__init__()

        self.is_causal = is_causal

        self.attention = MultiHeadSelfAttention(
            embed_dim, n_head, self.is_causal, positional_encoder_type)
        self.dropout_attention = torch.nn.Dropout(dropout_rate)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.feedforward = PointwiseFeedforward(
            embed_dim, hidden_size, dropout_rate)
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)

    @profile_range("transformer_decoder_block_forward", domain="world_machine")
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Masked Multi-Head Attention
        y1 = self.layer_norm1(x)
        y1 = self.dropout_attention(self.attention(y1))

        # Add
        y1 = x+y1

        y2 = self.layer_norm2(y1)

        # Feed Forward
        y2 = self.feedforward(y2)

        # Add
        result = y1+y2

        return result
