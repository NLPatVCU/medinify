from medinify.process import *
import numpy as np
from gensim.models import KeyedVectors
import warnings

warnings.filterwarnings("ignore")


class EmbeddingsProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.w2v = None

    def get_features(self, dataset):
        dataset = remove_neutral(dataset)
        if not self.w2v:
            self.w2v = KeyedVectors.load_word2vec_format(dataset.word_embeddings)
        comments = dataset.data_table[dataset.text_column]
        embeddings = np.zeros((dataset.data_table[dataset.text_column].shape[0], 100))
        for i, comment in enumerate(comments):
            tokens = self.tokenize(comment)
            all_embeddings = []
            for token in tokens:
                try:
                    all_embeddings.append(self.w2v[token])
                except KeyError:
                    continue
            if len(all_embeddings) == 0:
                continue
            else:
                embeddings[i] = np.average(all_embeddings, axis=0)
        return embeddings

    def get_labels(self, dataset):
        dataset = remove_neutral(dataset)
        labels = BowProcessor().get_labels(dataset)
        return labels
