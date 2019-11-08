
from medinify.vectorizers import Vectorizer
from medinify.vectorizers.utils import find_embeddings
import numpy as np
from gensim.models import KeyedVectors
import warnings

warnings.filterwarnings("ignore")


class EmbeddingsVectorizer(Vectorizer):
    """
    The EmbeddingsVectorizer transforms text data into averaged
    word embeddings representation to be fed into classifier
    """
    nickname = 'embedding'

    def __init__(self):
        """
        Constructor for EmbeddingsVectorizer
        :attribute w2v: (gensim.models.word2vec) pretrained word embeddings
        """
        super().__init__()
        embeddings_file = find_embeddings()
        self.w2v = KeyedVectors.load_word2vec_format(embeddings_file)

    def get_features(self, dataset):
        """
        Transforms text from dataset into averaged word embeddings
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: (np.array) averaged embedding representations of texts
        """
        comments = dataset.data_table[dataset.text_column]
        embeddings = np.zeros((dataset.data_table[dataset.text_column].shape[0], self.w2v.vector_size))
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

