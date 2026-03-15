import numpy as np


def create_causal_mask(size):
    mask = np.triu(np.ones((size, size)) * -np.inf, k=1)
    return mask