import numpy as np


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):

    dk = Q.shape[-1]

    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(dk)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)

    output = np.matmul(attention_weights, V)

    return output