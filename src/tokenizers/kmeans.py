from .base import Tokenizer
from sklearn.cluster import KMeans


class KMeansTokenizer(Tokenizer):
    def __init__(self, train_data, *args, **kwargs):
        kmeans = KMeans(*args, **kwargs)
        kmeans.fit(train_data)

        self.kmeans = kmeans

    def tokenize(self, actions):
        tokens = self.kmeans.predict(actions)
        return tokens

    def detokenize(self, tokens):
        original = self.kmeans.cluster_centers_[tokens]
        return original

    def get_vocab_size(self):
        return len(self.kmeans.cluster_centers_)
