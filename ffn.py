import numpy as np


class FeedForward:
    def __init__(self, d_model=512, d_ff=2048):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((1, d_ff))

        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((1, d_model))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        hidden = self.relu(x @ self.W1 + self.b1)
        output = hidden @ self.W2 + self.b2
        return output