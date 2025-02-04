import torch

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim:int, n_head:int, is_causal:bool):
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(embed_dim, num_heads=n_head, bias=False, batch_first=True)

        self.is_causal = is_causal

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        context_size = x.shape[1]

        if self.is_causal:
            attention_mask = torch.nn.Transformer.generate_square_subsequent_mask(context_size)
            attention_out, _ = self.attention(x, x, x, need_weights=False, is_causal=True, attn_mask=attention_mask)
        else:
            attention_out, _ = self.attention(x, x, x, need_weights=False)

        return attention_out