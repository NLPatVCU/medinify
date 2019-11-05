from medinify.process import *
from torchtext.data import Field, LabelField, Example, BucketIterator
import torch
from torchtext.vocab import Vectors
from torchtext.data import Dataset as TorchtextDataset
from gensim.models import KeyedVectors
import numpy as np


class MatrixProcessor(Processor):

    nickname = 'matrix'

    def __init__(self):
        super().__init__()
        self.w2v = None
        """
        self.text_field = Field(tokenize=super().tokenize, dtype=torch.float64)
        self.label_field = LabelField(dtype=torch.float64)
        """

    def get_features(self, dataset):
        dataset = remove_neutral(dataset)
        dataset.data_table['length'] = dataset.data_table[dataset.text_column].str.len()
        dataset.data_table.sort_values(by='length', inplace=True)
        dataset.data_table = dataset.data_table.reset_index()
        del dataset.data_table['length']
        del dataset.data_table['index']

        if not self.w2v:
            self.w2v = KeyedVectors.load_word2vec_format(dataset.args['word_embeddings'])
        dataset.data_table['token'] = dataset.data_table.apply(lambda row: self.tokenize(row[dataset.text_column]), axis=1)

        matrices = dataset.data_table.apply(lambda x: self.tokens_to_matrix(x['token']), axis=1)
        dataset.data_table['matrices'] = matrices
        return matrices

    def get_labels(self, dataset):
        dataset = remove_neutral(dataset)
        labels = BowProcessor().get_labels(dataset)
        return labels

    def tokens_to_matrix(self, tokens):
        assert self.w2v, 'No word embeddings specified'
        matrix = np.zeros((len(tokens), 100))
        for i, token in enumerate(tokens):
            try:
                matrix[i] = self.w2v[token]
            except KeyError:
                continue
        return matrix

    """
    @staticmethod
    def unpack_samples(loader):
        labels = []
        samples = []
        for sample in loader:
            labels.extend([x.item() for x in list(sample.label.to(torch.int64))])
            samples.append(sample)
        return labels, samples
    """

