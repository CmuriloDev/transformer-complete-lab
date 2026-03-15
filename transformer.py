import numpy as np
from encoder import EncoderBlock
from decoder import DecoderBlock


class Transformer:
    def __init__(self, vocab_size=5000, d_model=512):

        self.encoder = EncoderBlock(d_model)
        self.decoder = DecoderBlock(d_model)

        self.linear = np.random.randn(d_model, vocab_size) * 0.01

    def softmax(self, x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=-1, keepdims=True)

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, y, encoder_output, mask):
        return self.decoder.forward(y, encoder_output, mask)

    def project(self, x):
        logits = x @ self.linear
        return self.softmax(logits)