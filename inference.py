import numpy as np


def generate_sequence(model, encoder_input, start_token=1, eos_token=0, max_len=10):

    encoder_output = model.encode(encoder_input)

    generated = [start_token]

    for _ in range(max_len):

        decoder_input = np.random.randn(1, len(generated), 512)

        decoder_output = model.decode(decoder_input, encoder_output)

        probs = model.project(decoder_output[:, -1, :])

        next_token = np.argmax(probs)

        generated.append(int(next_token))

        if next_token == eos_token:
            break

    return generated