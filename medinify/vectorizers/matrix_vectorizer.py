
from medinify.vectorizers import Vectorizer
from medinify.vectorizers.utils import find_embeddings
from gensim.models import KeyedVectors
import numpy as np


class MatrixVectorizer(Vectorizer):
    """
    The EmbeddingsVectorizer transforms text data into arrays of indices
    and provides functionality for generating an embedding lookup table
    When indices arrays are fed through lookup table (using pytorch embedding layer),
    creates matrix for word embeddings
    """
    nickname = 'matrix'

    def __init__(self):
        """
        Constructor for MatrixVectorizer
        :attribute w2v: (gensim.models.word2vec) pretrained word embeddings
        :attribute index_to_word: (list[str]) list of words in embeddings vocab,
            search using .index() to get from token to index
        """
        super().__init__()
        embeddings_file = find_embeddings()
        self.w2v = KeyedVectors.load_word2vec_format(embeddings_file)
        self.index_to_word = self.w2v.index2word

    def get_features(self, dataset):
        """
        Transforms text from dataset into arrays of indices
        Sorts Dataset based on length of indices array, because this data
        will have to be padded later (when feeding through a network) and sorting
        will reduce the amount of padding required
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: (np.array) arrays of indices in lookup table of embeddings for texts
        """
        tokens = dataset.data_table.apply(lambda row: self.tokenize(row[dataset.text_column]), axis=1)
        indices = tokens.apply(lambda row: self.tokens_to_indices(row))
        dataset.data_table['indices'] = indices
        dataset.data_table['len'] = dataset.data_table.apply(lambda row: len(row['indices']), axis=1)
        dataset.data_table.sort_values('len', inplace=True)
        dataset.data_table = dataset.data_table.loc[dataset.data_table['len'] > 3]
        dataset.data_table = dataset.data_table.drop('len', axis=1)
        return dataset.data_table['indices']

    def tokens_to_indices(self, tokens):
        """
        Transforms list of tokens into an array of indices
        :param tokens: (list[str]) tokens
        :return: (np.array) indices from/for lookup table
        """
        indices = np.zeros(len(tokens), dtype=int)
        for i, token in enumerate(tokens):
            try:
                indices[i] = self.index_to_word.index(token) + 1
            except ValueError:
                continue
        return indices

    def indices_to_tokens(self, indices):
        """
        Transforms array of indices into an array of tokens
        :param indices: (np.array) indices from/for lookup table
        :return: (np.array) tokens
        """
        tokens = np.empty((len(indices)), dtype=object)
        for i, index in enumerate(indices):
            if index != 0:
                tokens[i] = self.index_to_word[index - 1]
        return tokens



