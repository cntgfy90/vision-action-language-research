import numpy as np
from transformers import BertTokenizer
from .base import Tokenizer

# BERT specific max length
MAX_LENGTH = 512


class BERTTokenizer(Tokenizer):
    def __init__(self):
        self.bert = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, actions):
        # convert actions to quantized text
        action_texts = []
        for action in actions:
            rounded = [round(x, 1) for x in action]
            text = " ".join([f"action_{i}_{val}" for i, val in enumerate(rounded)])
            action_texts.append(text)

        # join all actions into one string
        full_text = " ".join(action_texts)

        tokens = self.bert.encode(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=MAX_LENGTH,
            return_overflowing_tokens=True,
        )

        return np.array(tokens)

    def detokenize(self):
        pass

    def get_vocab_size(self):
        return self.bert.vocab_size
