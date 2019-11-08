
from medinify.vectorizers import Vectorizer
from sklearn.feature_extraction.text import CountVectorizer


class BowVectorizer(Vectorizer):
    """
    The BowVectorizer (Bag-of-Words Vectorizer) transforms text data into
    bag-of-word representation to be fed into classifier
    """
    nickname = 'bow'

    def __init__(self):
        """
        Constructor for BowVectorizer
        :attribute vectorizer: (CountVectorizer) transforms text into bag-of-words
        """
        super().__init__()
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize)

    def get_features(self, dataset):
        """
        Transforms text from dataset into bag-of-words
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: (scipy.sparse.csr_matrix) bag-of-words representations of texts
        """
        try:
            self.vectorizer.vocabulary_
        except AttributeError:
            self.vectorizer.fit(dataset.data_table[dataset.text_column])
        count_vectors = self.vectorizer.transform(dataset.data_table[dataset.text_column])
        return count_vectors

