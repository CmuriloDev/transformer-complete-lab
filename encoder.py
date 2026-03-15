import numpy as np
from attention import scaled_dot_product_attention
from ffn import FeedForward
from add_norm import AddNorm


class EncoderBlock:
    def __init__(self, d_model=512):
        self.ffn = FeedForward(d_model)
        self.add_norm1 = AddNorm()
        self.add_norm2 = AddNorm()

    def forward(self, x):
        Q = x
        K = x
        V = x

        attention_output = scaled_dot_product_attention(Q, K, V, mask=None)

        x = self.add_norm1.forward(x, attention_output)

        ffn_output = self.ffn.forward(x)

        x = self.add_norm2.forward(x, ffn_output)

        return x