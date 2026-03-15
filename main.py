import numpy as np
from transformer import Transformer
from inference import generate_sequence


def main():

    print("TRANSFORMER COMPLETE LAB")

    encoder_input = np.random.randn(1, 5, 512)

    model = Transformer()

    sequence = generate_sequence(model, encoder_input)

    print("Generated tokens:", sequence)


if __name__ == "__main__":
    main()