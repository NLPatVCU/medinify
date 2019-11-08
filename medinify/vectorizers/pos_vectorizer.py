
from medinify.vectorizers import Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from medinify.vectorizers.utils import get_pos_list


class PosVectorizer(Vectorizer):
    """
    The PosVectorizer (Part-of-Speech Vectorizer) transforms text data into bag-of-word
    representations and specified parts of speech removed to be fed into classifier
    """
    nickname = 'pos'

    def __init__(self, pos_list=None):
        """
        Constructor for PosVectorizer
        :attribute vectorizer: (CountVectorizer) transforms text into bag-of-words
        :attribute pos_list: (list[str]) list of parts for speech to remove
        """
        super().__init__()
        self.vectorizer = CountVectorizer(tokenizer=self.pos_tokenize)
        with open('./data/pos_tags', 'r') as f:
            valid_tags = set(f.read().splitlines())
        if not pos_list:
            pos_list = get_pos_list()
        for pos in pos_list:
            assert pos in valid_tags, '%s not a valid part of speech tag' % pos
        self.pos_list = pos_list

    def get_features(self, dataset):
        """
        Transforms text from dataset into bag-of-words with parts of speech removed
        :param dataset: (Dataset) dataset containing data to be Vectorized
        :return: (scipy.sparse.csr_matrix) bag-of-words representations of texts
        """
        try:
            self.vectorizer.vocabulary_
        except AttributeError:
            self.vectorizer.fit(dataset.data_table[dataset.text_column])
        count_vectors = self.vectorizer.transform(dataset.data_table[dataset.text_column])
        return count_vectors

    def pos_tokenize(self, text):
        """
        Tokenizes and removes parts of speech
        :param text: text to be tokenized
        :return: tokens
        """
        tokens = [token.orth_ for token in self.nlp(text.lower())
                  if token.orth_ not in self.stops and not token.is_punct | token.is_space
                  and token.pos_ not in self.pos_list]
        return tokens

