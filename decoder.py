import numpy as np
from attention import scaled_dot_product_attention
from ffn import FeedForward
from add_norm import AddNorm
from mask import create_causal_mask


class DecoderBlock:
    def __init__(self, d_model=512):
        self.ffn = FeedForward(d_model)

        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()
        self.add_norm3 = AddNorm()

    def forward(self, y, encoder_output):

        seq_len = y.shape[1]
        mask = create_causal_mask(seq_len)

        Q = y
        K = y
        V = y

        masked_attention = scaled_dot_product_attention(Q, K, V, mask)

        y = self.add_norm1.forward(y, masked_attention)

        Q = y
        K = encoder_output
        V = encoder_output

        cross_attention = scaled_dot_product_attention(Q, K, V, mask=None)

        y = self.add_norm2.forward(y, cross_attention)

        ffn_output = self.ffn.forward(y)

        y = self.add_norm3.forward(y, ffn_output)

        return y