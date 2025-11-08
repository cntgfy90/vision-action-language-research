import numpy as np
from .base import Tokenizer
from tokenizers import Tokenizer as VendorTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer(Tokenizer):
    def __init__(self, train_data, precision=1000, vocab_size=100, *args, **kwargs):
        self.precision = precision

        tokenizer = VendorTokenizer(BPE(*args, **kwargs))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"], min_frequency=2
        )

        sequences = self._continuous_to_bpe_format(train_data)
        tokenizer.train_from_iterator(sequences, trainer)

        self.bpe = tokenizer

    def tokenize(self, actions):
        sequences = self._continuous_to_bpe_format(actions)

        all_bpe_tokens = []

        for seq in sequences:
            encoding = self.bpe.encode(seq)
            bpe_tokens = encoding.ids
            all_bpe_tokens.extend(bpe_tokens)

        return all_bpe_tokens

    def detokenize(self):
        raise NotImplementedError("BPE cannot reconstruct continuous actions")

    def get_vocab_size(self):
        return self.bpe.get_vocab_size()

    def _continuous_to_bpe_format(self, actions):
        """Convert continuous actions to high-precision discrete sequences"""
        sequences = []
        for action in actions:
            dim_tokens = []
            for dim, value in enumerate(action):
                # scale to integer and create token
                int_value = int(value * self.precision)
                dim_tokens.append(f"d{dim}_v{int_value}")
            sequences.append(" ".join(dim_tokens))
        return sequences
