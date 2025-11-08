import numpy as np

BITS_PER_FLOAT = 32


def measure_compression(tokenizer, actions):
    tokens = tokenizer.tokenize(actions)

    original_bits = actions.shape[1] * BITS_PER_FLOAT * len(actions)
    compressed_bits = np.ceil(np.log2(tokenizer.get_vocab_size())) * len(tokens)
    # bit compression ratio
    compression_ratio = original_bits / compressed_bits

    token_changes = np.sum(tokens[1:] != tokens[:-1])
    # run-length encoding potential
    max_rle_compression = len(tokens) / (token_changes + 1)

    return {
        "compression_ratio": compression_ratio,
        "max_rle_compression": max_rle_compression,
    }
