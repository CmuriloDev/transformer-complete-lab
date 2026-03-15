import numpy as np

class AddNorm:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + self.epsilon)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)