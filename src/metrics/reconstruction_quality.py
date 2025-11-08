import numpy as np


def evaluate_reconstruction_quality(tokenizer, actions):
    """
    Evaluate how well tokenizer reconstructs original actions
    """
    tokens = tokenizer.tokenize(actions)

    reconstructed_actions = tokenizer.detokenize(tokens)

    # calculate reconstruction errors
    mse = np.mean((actions - reconstructed_actions) ** 2)
    mae = np.mean(np.abs(actions - reconstructed_actions))

    # per-dimension errors
    dim_mse = np.mean((actions - reconstructed_actions) ** 2, axis=0)
    dim_mae = np.mean(np.abs(actions - reconstructed_actions), axis=0)

    return {
        "mse": mse,
        "mae": mae,
        "dim_mse": dim_mse,
        "dim_mae": dim_mae,
        "reconstructed_actions": reconstructed_actions,
    }
