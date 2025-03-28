from .alibi import AlibiPositionalEncoder, LearnableAlibiPositionalEncoder
from .positional_encoder import PositionalEncoder
from .sine import SinePositionalEncoder


def create_positional_encoder(encoder_type: str | None, embed_dim: int, sequence_size: int, n_head: int = 0) -> PositionalEncoder:
    encoder_class = None

    if encoder_type is None:
        encoder_class = PositionalEncoder
    elif encoder_type == "sine":
        encoder_class = SinePositionalEncoder
    elif encoder_type == "alibi":
        encoder_class = AlibiPositionalEncoder
    elif encoder_type == "learnable_alibi":
        encoder_class = LearnableAlibiPositionalEncoder
    else:
        raise ValueError("Invalid positional encoder type!")

    return encoder_class(embed_dim, sequence_size, n_head)
