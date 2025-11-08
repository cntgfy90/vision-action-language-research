from abc import ABC, abstractmethod


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def detokenize(self):
        pass

    @abstractmethod
    def get_vocab_size(self):
        pass
