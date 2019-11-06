
from medinify.process import *
from gensim.models import KeyedVectors
import numpy as np


class MatrixProcessor(Processor):

    nickname = 'matrix'

    def __init__(self):
        super().__init__()
        embeddings_file = find_embeddings()
        self.w2v = KeyedVectors.load_word2vec_format(embeddings_file)
        self.index_to_word = self.w2v.index2word

    def get_features(self, dataset):
        tokens = dataset.data_table.apply(lambda row: self.tokenize(row[dataset.text_column]), axis=1)
        indices = tokens.apply(lambda row: self.tokens_to_indices(row))
        dataset.data_table['indices'] = indices
        dataset.data_table['len'] = dataset.data_table.apply(lambda row: len(row['indices']), axis=1)
        dataset.data_table.sort_values('len', inplace=True)
        dataset.data_table = dataset.data_table.loc[dataset.data_table['len'] > 3]
        dataset.data_table = dataset.data_table.drop('len', axis=1)
        return dataset.data_table['indices']

    def tokens_to_indices(self, tokens):
        indices = np.zeros(len(tokens), dtype=int)
        for i, token in enumerate(tokens):
            try:
                indices[i] = self.index_to_word.index(token) + 1
            except ValueError:
                continue
        return indices

    def indices_to_tokens(self, indices):
        tokens = np.empty((len(indices)), dtype=object)
        for i, index in enumerate(indices):
            if index != 0:
                tokens[i] = self.index_to_word[index - 1]
        return tokens

    def get_lookup_table(self):
        index_to_word = self.w2v.index2word
        lookup_table = np.zeros((len(index_to_word) + 1, self.w2v.vector_size))
        for i, word in enumerate(index_to_word):
            lookup_table[i + 1] = self.w2v[word]
        return lookup_table


