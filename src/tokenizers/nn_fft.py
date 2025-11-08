import random
import torch
import torch.nn as nn
import numpy as np
from scipy.fft import fft
from .base import Tokenizer


class FFTTokenizer(Tokenizer):
    def __init__(
        self, vocab_size=512, fft_components=32, encoding_dim=32, device="cpu", seed=42
    ):
        self.vocab_size = vocab_size
        self.fft_components = fft_components
        self.encoder = None
        self.encoding_dim = encoding_dim
        self.decoder = None
        self.device = device
        self.seed = seed

    def train(self, actions, epochs):
        self._set_seeds(self.seed)
        fft_compressed = self._fft_compress(actions)
        self._train_neural_components(fft_compressed, epochs=epochs)

    def tokenize(self, actions):
        compressed = self._fft_compress(actions)

        with torch.no_grad():
            compressed_tensor = torch.FloatTensor(compressed).to(self.device)
            neural_codes = self.encoder(compressed_tensor)

        tokens = []
        for code in neural_codes:
            token = int(abs(torch.sum(code).item()) * 1000) % self.vocab_size
            tokens.append(token)

        return np.array(tokens)

    def detokenize(self):
        pass

    def get_vocab_size(self):
        return self.vocab_size

    def _fft_compress(self, actions):
        freq_domain = fft(actions, axis=0)
        magnitudes = np.abs(freq_domain)

        compressed_seq = np.zeros_like(actions)
        for i in range(actions.shape[1]):
            top_indices = np.argsort(magnitudes[:, i])[-self.fft_components :]
            # keep only real part for neural network processing
            compressed_seq[top_indices, i] = freq_domain[top_indices, i].real

        return compressed_seq

    def _train_neural_components(self, compressed_actions, epochs):
        input_dim = compressed_actions.shape[1]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, self.encoding_dim)
        )
        self.encoder.to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )
        self.decoder.to(self.device)

        # should be possible to passed dynamically
        optimizer = torch.optim.Adam(
            # learning rate should be adjustable
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001,
        )

        for epoch in range(epochs):
            x = torch.FloatTensor(compressed_actions).to(self.device)
            z = self.encoder(x)
            x_recon = self.decoder(z)
            loss = torch.mean((x_recon - x) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

    def _set_seeds(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # multi-GPU
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # reproducible CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
